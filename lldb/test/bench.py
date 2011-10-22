#!/usr/bin/env python

"""
A simple bench runner which delegates to the ./dotest.py test driver to run the
benchmarks defined in the list named 'benches'.

You need to hand edit 'benches' to modify/change the command lines passed to the
test driver.

Use the following to get only the benchmark results in your terminal output:

    ./bench.py 2>&1 | grep -P '^lldb.*benchmark:'
"""

import os, sys
import re

# dotest.py invocation with no '-e exe-path' uses lldb as the inferior program,
# unless there is a mentioning of custom executable program.
benches = [
    # Measure startup delays creating a target and setting a breakpoint at main.
    './dotest.py -v +b -n -p TestStartupDelays.py',

    # Measure 'frame variable' response after stopping at Driver::MainLoop().
    './dotest.py -v +b -x "-F Driver::MainLoop()" -n -p TestFrameVariableResponse.py',

    # Measure stepping speed after stopping at Driver::MainLoop().
    './dotest.py -v +b -x "-F Driver::MainLoop()" -n -p TestSteppingSpeed.py',

    # Measure expression cmd response with a simple custom executable program.
    './dotest.py +b -n -p TestExpressionCmd.py',

    # Attach to a spawned lldb process then run disassembly benchmarks.
    './dotest.py -v +b -n -p TestDoAttachThenDisassembly.py'
]

def main():
    """Read the items from 'benches' and run the command line one by one."""
    print "Starting bench runner...."

    for command in benches:
        print "Running %s" % (command)
        os.system(command)

    print "Bench runner done."

if __name__ == '__main__':
    main()
