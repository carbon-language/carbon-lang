#!/usr/bin/env python

from __future__ import print_function

import os
import re
import select
import subprocess
import sys

# Wrapping this rather than os.devnull since os.devnull does not seeem to implement 'write'.
class NullWriter(object):
    def write (self, *args):
        pass
    def writelines (self, *args):
        pass
    def close (self, *args):
        pass

def find_makefile_dirs():
    makefile_dirs = []
    for root, dirs, files in os.walk("."):
        for file in files:
            if file == "Makefile":
                makefile_dirs.append(root)
    return makefile_dirs

_TESTDIR_RELATIVE_REGEX = re.compile(r"^([^/:]+:\d+:)")
_COMBINER_TERMINATION_REGEX = re.compile(r"^.+FAILED.+$")

def filter_run_line(sub_expr, line):
    return _TESTDIR_RELATIVE_REGEX.subn(sub_expr, line)

def line_combine_printer(file, previous_data, new_line_subn_result):
    (accumulated_line, combine_lines_left) = previous_data
    (incoming_line, sub_match_count) = new_line_subn_result

    if sub_match_count > 0:
        # New line was a match.  Don't print yet, start an accumulation.
        if len(accumulated_line) > 0:
            # Flush anything previously there.
            print(accumulated_line, file=file)
        return (incoming_line + ": ", 3)
    else:
        # If we're combining and incoming is a "[  FAILED ]" line, we've gone too far on a combine.
        if (len(accumulated_line) > 0) and _COMBINER_TERMINATION_REGEX.match(incoming_line):
            # Stop the combine.
            print(accumulated_line, file=file)
            print(incoming_line, file=file)
            return ("", 0)

        if len(accumulated_line) > 0:
            if accumulated_line[-2:] != ": ":
                # Need to add a comma
                new_line = accumulated_line + ", " + incoming_line
            else:
                new_line = accumulated_line + incoming_line
        else:
            new_line = incoming_line

        remaining_count = combine_lines_left - 1
        if remaining_count > 0:
            return (new_line, remaining_count)
        else:
            # Time to write it out.
            if len(new_line) > 0:
                print(new_line, file=file)
            return ("", 0)

def call_make(makefile_dir, extra_args=None, stdout=sys.stdout, stderr=sys.stderr):
    command = ["make", "-C", makefile_dir]
    if extra_args:
        command.extend(extra_args)

    # Replace the matched no-directory filename with one where the makefile directory is prepended.
    sub_expr = makefile_dir + r"/\1";

    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    stdout_data = ("", 0)
    stderr_data = ("", 0)

    while True:
        reads = [proc.stdout.fileno(), proc.stderr.fileno()]
        select_result = select.select(reads, [], [])

        for fd in select_result[0]:
            if fd == proc.stdout.fileno():
                line = proc.stdout.readline()
                stdout_data = line_combine_printer(stdout, stdout_data, filter_run_line(sub_expr, line.rstrip()))
            elif fd == proc.stderr.fileno():
                line = proc.stderr.readline()
                stderr_data = line_combine_printer(stderr, stderr_data, filter_run_line(sub_expr, line.rstrip()))

        proc_retval = proc.poll()
        if proc_retval != None:
            # Process stopped.  Drain output before finishing up.

            # Drain stdout.
            while True:
                line = proc.stdout.readline().rstrip()
                if line and len(line) > 0:
                    stdout_data = line_combine_printer(stdout, stdout_data, filter_run_line(sub_expr, line))
                else:
                    break

            # Drain stderr.
            while True:
                line = proc.stderr.readline().rstrip()
                if line and len(line) > 0:
                    stderr_data = line_combine_printer(stderr, stderr_data, filter_run_line(sub_expr, line))
                else:
                    break

            return proc_retval


global_retval = 0

do_clean_only = ('clean' in sys.argv)

for makefile_dir in find_makefile_dirs():
    # If we're not only cleaning, we do the normal build.
    if not do_clean_only:
        print("making: {}".format(makefile_dir))
        retval = call_make(makefile_dir)
        # Remember any errors that happen here.
        if retval != 0:
            global_retval = retval

    # Now clean
    call_make(makefile_dir, ['clean'], stdout=NullWriter(), stderr=NullWriter())
        
sys.exit(global_retval)
