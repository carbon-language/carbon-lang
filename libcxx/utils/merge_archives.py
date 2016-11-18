#!/usr/bin/env python
#===----------------------------------------------------------------------===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is dual licensed under the MIT and the University of Illinois Open
# Source Licenses. See LICENSE.TXT for details.
#
#===----------------------------------------------------------------------===##

from argparse import ArgumentParser
import distutils.spawn
import glob
import tempfile
import os
import shutil
import subprocess
import signal
import sys

temp_directory_root = None
def exit_with_cleanups(status):
    if temp_directory_root is not None:
        shutil.rmtree(temp_directory_root)
    sys.exit(status)

def print_and_exit(msg):
    sys.stderr.write(msg + '\n')
    exit_with_cleanups(1)

def diagnose_missing(file):
    if not os.path.exists(file):
        print_and_exit("input '%s' does not exist" % file)


def execute_command(cmd, cwd=None):
    """
    Execute a command, capture and return its output.
    """
    kwargs = {
        'stdin': subprocess.PIPE,
        'stdout': subprocess.PIPE,
        'stderr': subprocess.PIPE,
        'cwd': cwd
    }
    p = subprocess.Popen(cmd, **kwargs)
    out, err = p.communicate()
    exitCode = p.wait()
    if exitCode == -signal.SIGINT:
        raise KeyboardInterrupt
    return out, err, exitCode


def execute_command_verbose(cmd, cwd=None, verbose=False):
    """
    Execute a command and print its output on failure.
    """
    out, err, exitCode = execute_command(cmd, cwd=cwd)
    if exitCode != 0 or verbose:
        report = "Command: %s\n" % ' '.join(["'%s'" % a for a in cmd])
        if exitCode != 0:
            report += "Exit Code: %d\n" % exitCode
        if out:
            report += "Standard Output:\n--\n%s--" % out
        if err:
            report += "Standard Error:\n--\n%s--" % err
        if exitCode != 0:
            report += "\n\nFailed!"
        sys.stderr.write('%s\n' % report)
        if exitCode != 0:
            exit_with_cleanups(exitCode)

def main():
    parser = ArgumentParser(
        description="Merge multiple archives into a single library")
    parser.add_argument(
        '-v', '--verbose', dest='verbose', action='store_true', default=False)
    parser.add_argument(
        '-o', '--output', dest='output', required=True,
        help='The output file. stdout is used if not given',
        type=str, action='store')
    parser.add_argument(
        'archives', metavar='archives',  nargs='+',
        help='The archives to merge')

    args = parser.parse_args()

    ar_exe = distutils.spawn.find_executable('ar')
    if not ar_exe:
        print_and_exit("failed to find 'ar' executable")

    if len(args.archives) < 2:
        print_and_exit('fewer than 2 inputs provided')
    archives = []
    for ar in args.archives:
        diagnose_missing(ar)
        # Make the path absolute so it isn't affected when we change the PWD.
        archives += [os.path.abspath(ar)]

    if not os.path.exists(os.path.dirname(args.output)):
        print_and_exit("output path doesn't exist: '%s'" % args.output)

    global temp_directory_root
    temp_directory_root = tempfile.mkdtemp('.libcxx.merge.archives')

    for arc in archives:
        execute_command_verbose([ar_exe, '-x', arc], cwd=temp_directory_root,
                                verbose=args.verbose)

    files = glob.glob(os.path.join(temp_directory_root, '*.o'))
    if not files:
        print_and_exit('Failed to glob for %s' % glob_path)
    cmd = [ar_exe, '-qc', args.output] + files
    execute_command_verbose(cmd, cwd=temp_directory_root, verbose=args.verbose)


if __name__ == '__main__':
    main()
    exit_with_cleanups(0)
