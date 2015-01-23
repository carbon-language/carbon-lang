#!/usr/bin/env python

from __future__ import print_function

import os
import re
import select
import subprocess
import sys

# Wrapping this rather than os.devnull since os.devnull does not seem to implement 'write'.
class NullWriter(object):
    def write (self, *args):
        pass
    def writelines (self, *args):
        pass
    def close (self, *args):
        pass

# Find all "Makefile"-s in the current directory.
def find_makefile_dirs():
    makefile_dirs = []
    for root, dirs, files in os.walk("."):
        for file in files:
            if file == "Makefile":
                makefile_dirs.append(root)
    return makefile_dirs

# Test if line starts with <file name>:<line number> pattern
_TESTDIR_RELATIVE_REGEX = re.compile(r"^([^/:]+:\d+:)")

# Test if the line starts with the string "[ FAILED ]" (whitespace
# independent) what displayed on the end of a failed test case
_COMBINER_TERMINATION_REGEX = re.compile(r"^\[ *FAILED *\]")

# Prepends directory before each file name in line matching the regular
# expression "_TESTDIR_RELATIVE_REGEX" and returns the new value of line and the
# number of file names modified as a tuple.
def expand_file_name(directory, line):
    return _TESTDIR_RELATIVE_REGEX.subn(directory + r"/\1", line)

# Combine the failure report information from the output of gtest into a
# single line for better displaying inside IDEs
def line_combine_printer(file, previous_data, new_line_subn_result):
    (incoming_line, sub_match_count) = new_line_subn_result

    if sub_match_count > 0:
        # New line was a match for a file name. It means is the first line of
        # a failure report. Don't print yet, start an accumulation for more
        # info about the failure.
        if len(previous_data) > 0:
            # Flush anything previously accumulated (most likely part of the
            # previous failure report).
            print(previous_data, file=file)
        return incoming_line + ": "
    else:
        # If we're combining and incoming is a "[ FAILED ]" line then we have
        # to stop combining now.
        if (len(previous_data) > 0) and _COMBINER_TERMINATION_REGEX.match(incoming_line):
            # Stop the combine and print out its data and the FAIL line also.
            print(previous_data, file=file)
            print(incoming_line, file=file)
            return ""

        if len(previous_data) > 0:
            # Previous data is available what means we are currently
            # accumulating a failure report. Append this line to it.
            if len(previous_data) >= 2 and previous_data[-2:] != ": ":
                return previous_data + ", " + incoming_line
            else:
                return previous_data + incoming_line
        else:
            # No previous data and don't have to start new accumulation. Just
            # print the incoming line if it is not empty.
            if len(incoming_line) > 0:
                print(incoming_line, file=file)
            return ""

def call_make(makefile_dir, extra_args=None, stdout=sys.stdout, stderr=sys.stderr):
    command = ["make", "-C", makefile_dir]
    if extra_args:
        command.extend(extra_args)

    # Execute make as a new subprocess
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    stdout_data, stderr_data = "", ""

    while True:
        reads = [proc.stdout.fileno(), proc.stderr.fileno()]
        select_result = select.select(reads, [], [])

        # Copy the currently available output from make to the standard output
        # streams (stdout or stderr)
        for fd in select_result[0]:
            if fd == proc.stdout.fileno():
                line = proc.stdout.readline()
                stdout_data = line_combine_printer(stdout, stdout_data, expand_file_name(makefile_dir, line.rstrip()))
            elif fd == proc.stderr.fileno():
                line = proc.stderr.readline()
                stderr_data = line_combine_printer(stderr, stderr_data, expand_file_name(makefile_dir, line.rstrip()))

        proc_retval = proc.poll()
        if proc_retval != None:
            # Process stopped. Drain output before finishing up.

            # Drain stdout.
            while True:
                line = proc.stdout.readline()
                if line and len(line) > 0:
                    stdout_data = line_combine_printer(stdout, stdout_data, expand_file_name(makefile_dir, line.rstrip()))
                else:
                    break

            # Drain stderr.
            while True:
                line = proc.stderr.readline()
                if line and len(line) > 0:
                    stderr_data = line_combine_printer(stderr, stderr_data, expand_file_name(makefile_dir, line.rstrip()))
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
            print()
            print("-" * 80)
            print("Tests failed for Makefile in directory: %s" % makefile_dir)
            print("-" * 80)
            print()
            global_retval = retval

    # Now clean
    call_make(makefile_dir, ['clean'], stdout=NullWriter(), stderr=NullWriter())

if global_retval == 0:
    print()
    print("========================")
    print("| All tests are PASSED |")
    print("========================")
else:
    print()
    print("=========================================================")
    print("| Some of the test cases are FAILED with return code: %d |" % global_retval)
    print("=========================================================")
sys.exit(global_retval)
