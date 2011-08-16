#!/usr/bin/env python

"""
A simple utility to redo the failed/errored tests.

You need to specify the session directory in order for this script to locate the
tests which need to be re-run.

See also dotest.py, the test driver running the test suite.

Type:

./dotest.py -h

for help.
"""

import os, sys
import re

# If True, redo with no '-t' option for the test driver.
no_trace = False

# To be filled with the filterspecs found in the session logs.
redo_specs = []
comp_specs = set()
arch_specs = set()

def usage():
    print"""\
Usage: redo.py [-n] session_dir
where options:
-n : when running the tests, do not turn on trace mode, i.e, no '-t' option
     is passed to the test driver (this will run the tests faster)

and session_dir specifies the session directory which contains previously
recorded session infos for all the test cases which either failed or errored."""
    sys.exit(0)

def where(session_dir, test_dir):
    """Returns the full path to the session directory; None if non-existent."""
    abspath = os.path.abspath(session_dir)
    if os.path.isdir(abspath):
        return abspath

    session_dir_path = os.path.join(test_dir, session_dir)
    if os.path.isdir(session_dir_path):
        return session_dir_path

    return None

# This is the pattern for the line from the log file to redo a test.
# We want the filter spec.
filter_pattern = re.compile("^\./dotest\.py.*-f (.*)$")
comp_pattern = re.compile(" -C ([^ ]+) ")
arch_pattern = re.compile(" -A ([^ ]+) ")
def redo(suffix, dir, names):
    """Visitor function for os.path.walk(path, visit, arg)."""
    global redo_specs
    global comp_specs
    global arch_specs
    global filter_pattern
    global comp_pattern
    global arch_pattern

    for name in names:
        if name.endswith(suffix):
            #print "Find a log file:", name
            if name.startswith("Error") or name.startswith("Failure"):
                with open(os.path.join(dir, name), 'r') as log:
                    content = log.read()
                    for line in content.splitlines():
                        match = filter_pattern.match(line)
                        if match:
                            filterspec = match.group(1)
                            print "adding filterspec:", filterspec
                            redo_specs.append(filterspec)
                            comp = comp_pattern.search(line)
                            if comp:
                                comp_specs.add(comp.group(1))
                            arch = arch_pattern.search(line)
                            if arch:
                                arch_specs.add(arch.group(1))                                
            else:
                continue

def main():
    """Read the session directory and run the failed test cases one by one."""
    global no_trace
    global redo_specs

    if len(sys.argv) < 2 or len(sys.argv) > 3:
        usage()

    index = 1
    while index < len(sys.argv):
        if sys.argv[index].startswith('-'):
            # We should continue processing...
            pass
        else:
            # End of option processing.
            break

        if sys.argv[index] == '-n':
            no_trace = True
            index += 1

    session_dir = sys.argv[index]

    test_dir = sys.path[0]
    if not test_dir.endswith('test'):
        print "This script expects to reside in lldb's test directory."
        sys.exit(-1)

    #print "The test directory:", test_dir
    session_dir_path = where(session_dir, test_dir)

    #print "Session dir path:", session_dir_path
    os.chdir(test_dir)
    os.path.walk(session_dir_path, redo, ".log")

    filters = " -f ".join(redo_specs)
    compilers = (" -C %s" % "^".join(comp_specs)) if comp_specs else None
    archs = (" -A %s" % "^".join(arch_specs)) if arch_specs else None

    command = "./dotest.py %s %s -v %s -f " % (compilers if compilers else "",
                                               archs if archs else "",
                                               "" if no_trace else "-t")


    print "Running %s" % (command + filters)
    os.system(command + filters)

if __name__ == '__main__':
    main()
