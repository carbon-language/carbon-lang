#!/usr/bin/env python

"""
A simple testing framework for lldb using python's unit testing framework.

Tests for lldb are written as python scripts which take advantage of the script
bridging provided by LLDB.framework to interact with lldb core.

A specific naming pattern is followed by the .py script to be recognized as
a module which implements a test scenario, namely, Test*.py.

To specify the directories where "Test*.py" python test scripts are located,
you need to pass in a list of directory names.  By default, the current
working directory is searched if nothing is specified on the command line.
"""

import os
import sys
import time
import unittest2

#
# Global variables:
#

# The test suite.
suite = unittest2.TestSuite()

# Default verbosity is 0.
verbose = 0

# By default, search from the current working directory.
testdirs = [ os.getcwd() ]


def usage():
    print """
Usage: dotest.py [option] [args]
where options:
-h   : print this help message and exit (also --help)
-v   : do verbose mode of unittest framework

and:
args : specify a list of directory names to search for python Test*.py scripts
       if empty, search from the curret working directory, instead

Running of this script also sets up the LLDB_TEST environment variable so that
individual test cases can locate their supporting files correctly.
"""


def setupSysPath():
    """Add LLDB.framework/Resources/Python to the search paths for modules."""

    # Get the directory containing the current script.
    scriptPath = sys.path[0]
    if not scriptPath.endswith('test'):
        print "This script expects to reside in lldb's test directory."
        sys.exit(-1)

    os.environ["LLDB_TEST"] = scriptPath

    base = os.path.abspath(os.path.join(scriptPath, os.pardir))
    dbgPath = os.path.join(base, 'build', 'Debug', 'LLDB.framework',
                           'Resources', 'Python')
    relPath = os.path.join(base, 'build', 'Release', 'LLDB.framework',
                           'Resources', 'Python')

    lldbPath = None
    if os.path.isfile(os.path.join(dbgPath, 'lldb.py')):
        lldbPath = dbgPath
    elif os.path.isfile(os.path.join(relPath, 'lldb.py')):
        lldbPath = relPath

    if not lldbPath:
        print 'This script requires lldb.py to be in either ' + dbgPath,
        print ' or' + relPath
        sys.exit(-1)

    sys.path.append(lldbPath)
    sys.path.append(scriptPath)


def initTestdirs():
    """Initialize the list of directories containing our unittest scripts.

    '-h/--help as the first option prints out usage info and exit the program.
    """

    global verbose
    global testdirs

    if len(sys.argv) == 1:
        pass
    elif sys.argv[1].find('-h') != -1:
        # Print usage info and exit.
        usage()
        sys.exit(0)
    else:
        # Process possible verbose flag.
        index = 1
        if sys.argv[1].find('-v') != -1:
            verbose = 2
            index += 1

        # Gather all the dirs passed on the command line.
        if len(sys.argv) > index:
            testdirs = map(os.path.abspath, sys.argv[index:])


def visit(prefix, dir, names):
    """Visitor function for os.path.walk(path, visit, arg)."""

    global suite

    for name in names:
        if os.path.isdir(os.path.join(dir, name)):
            continue

        if '.py' == os.path.splitext(name)[1] and name.startswith(prefix):
            # We found a pattern match for our test case.  Add it to the suite.
            if not sys.path.count(dir):
                sys.path.append(dir)
            base = os.path.splitext(name)[0]
            suite.addTests(unittest2.defaultTestLoader.loadTestsFromName(base))


#
# Start the actions by first setting up the module search path for lldb,
# followed by initializing the test directories, and then walking the directory
# trees, while collecting the tests into our test suite.
#
setupSysPath()
initTestdirs()
for testdir in testdirs:
    os.path.walk(testdir, visit, 'Test')

# Now that we have loaded all the test cases, run the whole test suite.

# For the time being, let's bracket the test runner within the
# lldb.SBDebugger.Initialize()/Terminate() pair.
import lldb
lldb.SBDebugger.Initialize()

# Create a singleton SBDebugger in the lldb namespace.
lldb.DBG = lldb.SBDebugger.Create()

# Turn on logging for debugging purposes if ${LLDB_LOG} environment variable is
# is defined.  Use ${LLDB_LOG} to specify the log file.
ci = lldb.DBG.GetCommandInterpreter()
res = lldb.SBCommandReturnObject()
if ("LLDB_LOG" in os.environ):
    ci.HandleCommand(
        "log enable -f " + os.environ["LLDB_LOG"] + " lldb default", res)
    if not res.Succeeded():
        raise Exception('log enable failed (check your LLDB_LOG env variable...')

unittest2.TextTestRunner(verbosity=verbose).run(suite)

# Add some delay before calling SBDebugger.Terminate().
time.sleep(1)
lldb.SBDebugger.Terminate()
