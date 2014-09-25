#!/usr/bin/env python

from __future__ import print_function

import os
import re
import select
import subprocess
import sys

def find_makefile_dirs():
    makefile_dirs = []
    for root, dirs, files in os.walk("."):
        for file in files:
            if file == "Makefile":
                makefile_dirs.append(root)
    return makefile_dirs

_TESTDIR_RELATIVE_REGEX = re.compile(r"^([^/:]+:\d+:)")

def filter_run_line(sub_expr, line):
    return _TESTDIR_RELATIVE_REGEX.sub(sub_expr, line)
 
def call_make(makefile_dir, extra_args=None):
    command = ["make", "-C", makefile_dir]
    if extra_args:
        command.extend(extra_args)

    # Replace the matched no-directory filename with one where the makefile directory is prepended.
    sub_expr = makefile_dir + r"/\1";

    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    while True:
        reads = [proc.stdout.fileno(), proc.stderr.fileno()]
        select_result = select.select(reads, [], [])

        for fd in select_result[0]:
            if fd == proc.stdout.fileno():
                line = proc.stdout.readline()
                print(filter_run_line(sub_expr, line.rstrip()))
            elif fd == proc.stderr.fileno():
                line = proc.stderr.readline()
                print(filter_run_line(sub_expr, line.rstrip()), file=sys.stderr)

        proc_retval = proc.poll()
        if proc_retval != None:
            # Process stopped.  Drain output before finishing up.

            # Drain stdout.
            while True:
                line = proc.stdout.readline()
                if line:
                    print(filter_run_line(sub_expr, line.rstrip()))
                else:
                    break

            # Drain stderr.
            while True:
                line = proc.stderr.readline()
                if line:
                    print(filter_run_line(sub_expr, line.rstrip()), file=sys.stderr)
                else:
                    break

            return proc_retval


global_retval = 0
extra_args = None

if len(sys.argv) > 1:
    if sys.argv[1] == 'clean':
        extra_args = ['clean']

for makefile_dir in find_makefile_dirs():
    print("found makefile dir: {}".format(makefile_dir))
    retval = call_make(makefile_dir, extra_args)
    if retval != 0:
        global_retval = retval
        
sys.exit(global_retval)
