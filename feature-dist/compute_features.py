from numpy import array_split
import h5py
import paramiko
import yaml

import argparse
from collections import defaultdict
import errno
import getpass
import glob
import os
import sys
import tarfile
import tempfile
import fix_constraints

def read_conf(confpath):
    with open(confpath, 'r') as conffile:
         conf = yaml.safe_load(conffile)
    return conf

def read_logins(loginpath):
    with open(loginpath, 'r') as loginfile:
        loginconf = yaml.safe_load(loginfile)
    return defaultdict(lambda: None, loginconf)

def split_indices(datafile, n):
    if 'pred' not in datafile['0']:
        return array_split(range(len(datafile.keys()), n))
    else:
        # Create list of trajectories, where each element is a list of indices
        # for a particular trajectory.
        traj_indices = []
        curr_traj_indices = []
        for i in range(len(datafile)):
            if datafile[str(i)]['pred'][()] == str(i):
                if curr_traj_indices:
                    traj_indices.append(curr_traj_indices)
                curr_traj_indices = [i]
            else:
                curr_traj_indices.append(i)
        traj_indices.append(curr_traj_indices)
        split_traj_indices = array_split(traj_indices, n)
        grouped_indices = []
        for split_traj in split_traj_indices:
            grouped_indices.append([item for sublist in split_traj for item in sublist])
        print grouped_indices
        return grouped_indices

def split_data(datafname, outfolder, n):
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    print datafname
    with h5py.File(datafname, 'r') as datafile:
        num_examples = len(datafile.keys())
        grouped_indices = split_indices(datafile, n)
        for outfile_i, indices in enumerate(grouped_indices):
            outfname = os.path.join(outfolder, '{}.h5'.format(outfile_i))
            copy_indices(datafile, indices, outfname)

def copy_indices(datafile, indices, outfname):
    num_skipped = 0
    with h5py.File(outfname, 'w') as outfile:
        for i, index in enumerate(indices):
            # if datafile[str(index)]['action'][()].startswith('endstate'):
            #     num_skipped += 1
            #     continue
            outfile.copy(datafile[str(index)], str(i))
            if 'pred' in datafile[str(index)]: # Update 'pred' value
                if datafile[str(index)]['pred'][()] == str(index):
                    outfile[str(i)]['pred'][()] = str(i)
                else:
                    outfile[str(i)]['pred'][()] = str(i-1)

    print "Skipped {} endstates.".format(num_skipped)

def pack_payload(conf):
    info = conf['payload']
    path = info['path']
    fnames = [os.path.join(path, fname) for fname in os.listdir(path)]
    _, tarfname = tempfile.mkstemp(suffix='.tar.gz')
    with tarfile.open(name=tarfname, mode='w:gz', dereference=True) as tar:
        for fname in fnames:
            tar.add(fname, arcname=os.path.relpath(fname, path))
        for fileinfo in info['additional-files']:
            tar.add(fileinfo['path'], arcname=fileinfo['archive-name'])
    assert os.path.exists(tarfname)
    return tarfname

def add_to_end(final, new, i):
    # Assumes the keys in the file 'new' are consecutive integers starting
    # from zero
    for index in range(len(new)):
        final.copy(new[str(index)], str(i))
        i += 1
    return i

def rexists(sftp, path):
    try:
        sftp.stat(path)
    except IOError, e:
        if e.errno == errno.ENOENT:
            return False
        raise
    return True

def check_remote_overwrites(conf, logins=None, password=None):
    if logins is None:
        logins = defaultdict(lambda: None)
    hosts, paths = [], []
    for server in conf['servers']:
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        username = logins[server['host']]
        client.connect(server['host'], username=username, password=password)
        sftp = client.open_sftp()
        exists = rexists(sftp, server['path'])
        sftp.close()
        if exists:
            hosts.append(server['host'])
            paths.append(server['path'])
    return hosts, paths

def distribute_jobs(conf, logins=None, password=None, overwrite=False, has_constraints=False):
    if logins is None:
        logins = defaultdict(lambda: None)
    servers = conf['servers']
    problem_hosts, problem_paths = check_remote_overwrites(conf, logins=logins, password=password)
    if problem_hosts and not overwrite:
        for host, path in zip(problem_hosts, problem_paths):
            print 'ERROR: {} on {} already exists!'.format(path, host)
        print 'Terminating.'
        exit(1)
    num_jobs = sum(server['cores'] for server in servers)
    split_data(conf['datafile'], conf['splitsdir'], num_jobs)
    split_i = 0
    stdouts = []
    print 'Preparing payload...'
    payloadtar = pack_payload(conf)
    for server in servers:
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        username = logins[server['host']]
        client.connect(server['host'], username=username, password=password)
        if server['host'] in problem_hosts and overwrite:
            _, stdout, _ = client.exec_command('rm -rf {}'.format(server['path']))
            stdout.readlines()
        _, stdout, _ = client.exec_command('mkdir -p {}'.format(server['path']))
        stdout.readlines()
        sftp = client.open_sftp()
        print 'Copying scripts to {}...'.format(server['host'])
        remote_path = os.path.join(server['path'], os.path.basename(payloadtar))
        sftp.put(payloadtar, remote_path)
        _, stdout, _ = client.exec_command('tar -xzf {} -C {}'.format(remote_path, server['path']))
        stdout.readlines()
        print 'Copying data files to {}...'.format(server['host'])
        _, stdout, _ = client.exec_command('mkdir -p {}'.format(os.path.join(server['path'], 'splits')))
        stdout.readlines()
        for i in range(split_i, split_i + server['cores']):
            sftp.put(os.path.join(conf['splitsdir'], '{}.h5'.format(i)),
                     os.path.join(server['path'], 'splits', '{}.h5'.format(i)))
        split_i += server['cores']
        stdin, stdout, stderr = client.exec_command("python {}".format(os.path.join(server['path'], 'driver.py')))
        stdouts.append(stdout)
    print "Waiting for servers to finish..."
    print [stdout.channel.recv_exit_status() for stdout in stdouts]
    print "Done. Collecting results..."
    collect_results(conf, logins=logins, password=password, has_constraints=has_constraints)

def collect_results(conf, logins=None, password=None, has_constraints=False):
    if logins is None:
        logins = defaultdict(lambda: None)
    if not os.path.exists(conf['outfolder']):
        os.makedirs(conf['outfolder'])
    for server in conf['servers']:
        print '- Collecting results from {}'.format(server['host'])
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        username = logins[server['host']]
        client.connect(server['host'], username=username, password=password)
        sftp = client.open_sftp()
        fnames = sftp.listdir(os.path.join(server['path'], 'out'))
        for fname in fnames:
            sftp.get(os.path.join(server['path'], 'out', fname),
                     os.path.join(conf['outfolder'], fname))
    outfiles = glob.glob(os.path.join(conf['outfolder'], '*.h5'))
    outfiles.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    final_outfile = h5py.File(conf['outfile'], 'w')
    i = 0
    for outfile in outfiles:
        i = add_to_end(final_outfile, h5py.File(outfile, 'r'), i)
    final_outfile.close()
    if has_constraints:
        fix_constraints.make_slack_names_unique(conf['outfile'])
    print "Final results in {}.".format(conf['outfile'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('serverconf')
    parser.add_argument('loginconf', nargs='?', default='logins.yml')
    parser.add_argument('--overwrite', action='store_true', default=False)
    parser.add_argument('--collect_only', action='store_true', default=False)
    parser.add_argument('--has_constraints', action='store_true', default=False)
    args = parser.parse_args()
    logins = defaultdict(lambda: None)
    if os.path.exists(args.loginconf):
        print 'Using login information in {}.'.format(args.loginconf)
        logins = read_logins(args.loginconf)
    conf = read_conf(args.serverconf)
    pw = getpass.getpass()
    if args.collect_only:
        collect_results(conf, logins=logins, password=pw, has_constraints=args.has_constraints)
    else:
        distribute_jobs(conf, logins=logins, password=pw, overwrite=args.overwrite, has_constraints=args.has_constraints)
