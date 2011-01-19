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

Type:

./dotest.py -h

for available options.
"""

import os, signal, sys, time
import unittest2

class _WritelnDecorator(object):
    """Used to decorate file-like objects with a handy 'writeln' method"""
    def __init__(self,stream):
        self.stream = stream

    def __getattr__(self, attr):
        if attr in ('stream', '__getstate__'):
            raise AttributeError(attr)
        return getattr(self.stream,attr)

    def writeln(self, arg=None):
        if arg:
            self.write(arg)
        self.write('\n') # text-mode streams translate to \r\n if needed

#
# Global variables:
#

# The test suite.
suite = unittest2.TestSuite()

# By default, both command line and Python API tests are performed.
# Use @python_api_test decorator, defined in lldbtest.py, to mark a test as
# a Python API test.
dont_do_python_api_test = False

# By default, both command line and Python API tests are performed.
just_do_python_api_test = False

# The blacklist is optional (-b blacklistFile) and allows a central place to skip
# testclass's and/or testclass.testmethod's.
blacklist = None

# The dictionary as a result of sourcing blacklistFile.
blacklistConfig = {}

# The config file is optional.
configFile = None

# Test suite repeat count.  Can be overwritten with '-# count'.
count = 1

# The dictionary as a result of sourcing configFile.
config = {}

# Delay startup in order for the debugger to attach.
delay = False

# By default, failfast is False.  Use '-F' to overwrite it.
failfast = False

# The filter (testclass.testmethod) used to admit tests into our test suite.
filterspec = None

# If '-g' is specified, the filterspec is not exclusive.  If a test module does
# not contain testclass.testmethod which matches the filterspec, the whole test
# module is still admitted into our test suite.  fs4all flag defaults to True.
fs4all = True

# Ignore the build search path relative to this script to locate the lldb.py module.
ignore = False

# By default, we skip long running test case.  Use '-l' option to override.
skipLongRunningTest = True

# The regular expression pattern to match against eligible filenames as our test cases.
regexp = None

# By default, tests are executed in place and cleanups are performed afterwards.
# Use '-r dir' option to relocate the tests and their intermediate files to a
# different directory and to forgo any cleanups.  The directory specified must
# not exist yet.
rdir = None

# By default, recorded session info for errored/failed test are dumped into its
# own file under a session directory named after the timestamp of the test suite
# run.  Use '-s session-dir-name' to specify a specific dir name.
sdir_name = None

# Set this flag if there is any session info dumped during the test run.
sdir_has_content = False

# Default verbosity is 0.
verbose = 0

# By default, search from the current working directory.
testdirs = [ os.getcwd() ]

# Separator string.
separator = '-' * 70


def usage():
    print """
Usage: dotest.py [option] [args]
where options:
-h   : print this help message and exit (also --help)
-a   : don't do lldb Python API tests
       use @python_api_test to decorate a test case as lldb Python API test
+a   : just do lldb Python API tests
       do not specify both '-a' and '+a' at the same time
-b   : read a blacklist file specified after this option
-c   : read a config file specified after this option
       (see also lldb-trunk/example/test/usage-config)
-d   : delay startup for 10 seconds (in order for the debugger to attach)
-F   : failfast, stop the test suite on the first error/failure
-f   : specify a filter, which consists of the test class name, a dot, followed by
       the test method, to only admit such test into the test suite
       e.g., -f 'ClassTypesTestCase.test_with_dwarf_and_python_api'
-g   : if specified, the filterspec by -f is not exclusive, i.e., if a test module
       does not match the filterspec (testclass.testmethod), the whole module is
       still admitted to the test suite
-i   : ignore (don't bailout) if 'lldb.py' module cannot be located in the build
       tree relative to this script; use PYTHONPATH to locate the module
-l   : don't skip long running test
-p   : specify a regexp filename pattern for inclusion in the test suite
-r   : specify a dir to relocate the tests and their intermediate files to;
       the directory must not exist before running this test driver;
       no cleanup of intermediate test files is performed in this case
-s   : specify the name of the dir created to store the session files of tests
       with errored or failed status; if not specified, the test driver uses the
       timestamp as the session dir name
-t   : trace lldb command execution and result
-v   : do verbose mode of unittest framework
-w   : insert some wait time (currently 0.5 sec) between consecutive test cases
-#   : Repeat the test suite for a specified number of times

and:
args : specify a list of directory names to search for test modules named after
       Test*.py (test discovery)
       if empty, search from the curret working directory, instead

Examples:

This is an example of using the -f option to pinpoint to a specfic test class
and test method to be run:

$ ./dotest.py -f ClassTypesTestCase.test_with_dsym_and_run_command
----------------------------------------------------------------------
Collected 1 test

test_with_dsym_and_run_command (TestClassTypes.ClassTypesTestCase)
Test 'frame variable this' when stopped on a class constructor. ... ok

----------------------------------------------------------------------
Ran 1 test in 1.396s

OK

And this is an example of using the -p option to run a single file (the filename
matches the pattern 'ObjC' and it happens to be 'TestObjCMethods.py'):

$ ./dotest.py -v -p ObjC
----------------------------------------------------------------------
Collected 4 tests

test_break_with_dsym (TestObjCMethods.FoundationTestCase)
Test setting objc breakpoints using 'regexp-break' and 'breakpoint set'. ... ok
test_break_with_dwarf (TestObjCMethods.FoundationTestCase)
Test setting objc breakpoints using 'regexp-break' and 'breakpoint set'. ... ok
test_data_type_and_expr_with_dsym (TestObjCMethods.FoundationTestCase)
Lookup objective-c data types and evaluate expressions. ... ok
test_data_type_and_expr_with_dwarf (TestObjCMethods.FoundationTestCase)
Lookup objective-c data types and evaluate expressions. ... ok

----------------------------------------------------------------------
Ran 4 tests in 16.661s

OK

Running of this script also sets up the LLDB_TEST environment variable so that
individual test cases can locate their supporting files correctly.  The script
tries to set up Python's search paths for modules by looking at the build tree
relative to this script.  See also the '-i' option in the following example.

Finally, this is an example of using the lldb.py module distributed/installed by
Xcode4 to run against the tests under the 'forward' directory, and with the '-w'
option to add some delay between two tests.  It uses ARCH=x86_64 to specify that
as the architecture and CC=clang to specify the compiler used for the test run:

$ PYTHONPATH=/Xcode4/Library/PrivateFrameworks/LLDB.framework/Versions/A/Resources/Python ARCH=x86_64 CC=clang ./dotest.py -v -w -i forward

Session logs for test failures/errors will go into directory '2010-11-11-13_56_16'
----------------------------------------------------------------------
Collected 2 tests

test_with_dsym_and_run_command (TestForwardDeclaration.ForwardDeclarationTestCase)
Display *bar_ptr when stopped on a function with forward declaration of struct bar. ... ok
test_with_dwarf_and_run_command (TestForwardDeclaration.ForwardDeclarationTestCase)
Display *bar_ptr when stopped on a function with forward declaration of struct bar. ... ok

----------------------------------------------------------------------
Ran 2 tests in 5.659s

OK

The 'Session ...' verbiage is recently introduced (see also the '-s' option) to
notify the directory containing the session logs for test failures or errors.
In case there is any test failure/error, a similar message is appended at the
end of the stderr output for your convenience.

Environment variables related to loggings:

o LLDB_LOG: if defined, specifies the log file pathname for the 'lldb' subsystem
  with a default option of 'event process' if LLDB_LOG_OPTION is not defined.

o GDB_REMOTE_LOG: if defined, specifies the log file pathname for the
  'process.gdb-remote' subsystem with a default option of 'packets' if
  GDB_REMOTE_LOG_OPTION is not defined.
"""
    sys.exit(0)


def parseOptionsAndInitTestdirs():
    """Initialize the list of directories containing our unittest scripts.

    '-h/--help as the first option prints out usage info and exit the program.
    """

    global dont_do_python_api_test
    global just_do_python_api_test
    global blacklist
    global blacklistConfig
    global configFile
    global count
    global delay
    global failfast
    global filterspec
    global fs4all
    global ignore
    global skipLongRunningTest
    global regexp
    global rdir
    global sdir_name
    global verbose
    global testdirs

    if len(sys.argv) == 1:
        return

    # Process possible trace and/or verbose flag, among other things.
    index = 1
    while index < len(sys.argv):
        if sys.argv[index].startswith('-') or sys.argv[index].startswith('+'):
            # We should continue processing...
            pass
        else:
            # End of option processing.
            break

        if sys.argv[index].find('-h') != -1:
            usage()
        elif sys.argv[index].startswith('-a'):
            dont_do_python_api_test = True
            index += 1
        elif sys.argv[index].startswith('+a'):
            just_do_python_api_test = True
            index += 1
        elif sys.argv[index].startswith('-b'):
            # Increment by 1 to fetch the blacklist file name option argument.
            index += 1
            if index >= len(sys.argv) or sys.argv[index].startswith('-'):
                usage()
            blacklistFile = sys.argv[index]
            if not os.path.isfile(blacklistFile):
                print "Blacklist file:", blacklistFile, "does not exist!"
                usage()
            index += 1
            # Now read the blacklist contents and assign it to blacklist.
            execfile(blacklistFile, globals(), blacklistConfig)
            blacklist = blacklistConfig.get('blacklist')
        elif sys.argv[index].startswith('-c'):
            # Increment by 1 to fetch the config file name option argument.
            index += 1
            if index >= len(sys.argv) or sys.argv[index].startswith('-'):
                usage()
            configFile = sys.argv[index]
            if not os.path.isfile(configFile):
                print "Config file:", configFile, "does not exist!"
                usage()
            index += 1
        elif sys.argv[index].startswith('-d'):
            delay = True
            index += 1
        elif sys.argv[index].startswith('-F'):
            failfast = True
            index += 1
        elif sys.argv[index].startswith('-f'):
            # Increment by 1 to fetch the filter spec.
            index += 1
            if index >= len(sys.argv) or sys.argv[index].startswith('-'):
                usage()
            filterspec = sys.argv[index]
            index += 1
        elif sys.argv[index].startswith('-g'):
            fs4all = False
            index += 1
        elif sys.argv[index].startswith('-i'):
            ignore = True
            index += 1
        elif sys.argv[index].startswith('-l'):
            skipLongRunningTest = False
            index += 1
        elif sys.argv[index].startswith('-p'):
            # Increment by 1 to fetch the reg exp pattern argument.
            index += 1
            if index >= len(sys.argv) or sys.argv[index].startswith('-'):
                usage()
            regexp = sys.argv[index]
            index += 1
        elif sys.argv[index].startswith('-r'):
            # Increment by 1 to fetch the relocated directory argument.
            index += 1
            if index >= len(sys.argv) or sys.argv[index].startswith('-'):
                usage()
            rdir = os.path.abspath(sys.argv[index])
            if os.path.exists(rdir):
                print "Relocated directory:", rdir, "must not exist!"
                usage()
            index += 1
        elif sys.argv[index].startswith('-s'):
            # Increment by 1 to fetch the session dir name.
            index += 1
            if index >= len(sys.argv) or sys.argv[index].startswith('-'):
                usage()
            sdir_name = sys.argv[index]
            index += 1
        elif sys.argv[index].startswith('-t'):
            os.environ["LLDB_COMMAND_TRACE"] = "YES"
            index += 1
        elif sys.argv[index].startswith('-v'):
            verbose = 2
            index += 1
        elif sys.argv[index].startswith('-w'):
            os.environ["LLDB_WAIT_BETWEEN_TEST_CASES"] = 'YES'
            index += 1
        elif sys.argv[index].startswith('-#'):
            # Increment by 1 to fetch the repeat count argument.
            index += 1
            if index >= len(sys.argv) or sys.argv[index].startswith('-'):
                usage()
            count = int(sys.argv[index])
            index += 1
        else:
            print "Unknown option: ", sys.argv[index]
            usage()

    # Do not specify both '-a' and '+a' at the same time.
    if dont_do_python_api_test and just_do_python_api_test:
        usage()

    # Gather all the dirs passed on the command line.
    if len(sys.argv) > index:
        testdirs = map(os.path.abspath, sys.argv[index:])

    # If '-r dir' is specified, the tests should be run under the relocated
    # directory.  Let's copy the testdirs over.
    if rdir:
        from shutil import copytree, ignore_patterns

        tmpdirs = []
        for srcdir in testdirs:
            dstdir = os.path.join(rdir, os.path.basename(srcdir))
            # Don't copy the *.pyc and .svn stuffs.
            copytree(srcdir, dstdir, ignore=ignore_patterns('*.pyc', '.svn'))
            tmpdirs.append(dstdir)

        # This will be our modified testdirs.
        testdirs = tmpdirs

        # With '-r dir' specified, there's no cleanup of intermediate test files.
        os.environ["LLDB_DO_CLEANUP"] = 'NO'

        # If testdirs is ['test'], the make directory has already been copied
        # recursively and is contained within the rdir/test dir.  For anything
        # else, we would need to copy over the make directory and its contents,
        # so that, os.listdir(rdir) looks like, for example:
        #
        #     array_types conditional_break make
        #
        # where the make directory contains the Makefile.rules file.
        if len(testdirs) != 1 or os.path.basename(testdirs[0]) != 'test':
            # Don't copy the .svn stuffs.
            copytree('make', os.path.join(rdir, 'make'),
                     ignore=ignore_patterns('.svn'))

    #print "testdirs:", testdirs

    # Source the configFile if specified.
    # The side effect, if any, will be felt from this point on.  An example
    # config file may be these simple two lines:
    #
    # sys.stderr = open("/tmp/lldbtest-stderr", "w")
    # sys.stdout = open("/tmp/lldbtest-stdout", "w")
    #
    # which will reassign the two file objects to sys.stderr and sys.stdout,
    # respectively.
    #
    # See also lldb-trunk/example/test/usage-config.
    global config
    if configFile:
        # Pass config (a dictionary) as the locals namespace for side-effect.
        execfile(configFile, globals(), config)
        #print "config:", config
        #print "sys.stderr:", sys.stderr
        #print "sys.stdout:", sys.stdout


def setupSysPath():
    """Add LLDB.framework/Resources/Python to the search paths for modules."""

    global rdir
    global testdirs

    # Get the directory containing the current script.
    if "DOTEST_PROFILE" in os.environ and "DOTEST_SCRIPT_DIR" in os.environ:
        scriptPath = os.environ["DOTEST_SCRIPT_DIR"]
    else:
        scriptPath = sys.path[0]
    if not scriptPath.endswith('test'):
        print "This script expects to reside in lldb's test directory."
        sys.exit(-1)

    if rdir:
        # Set up the LLDB_TEST environment variable appropriately, so that the
        # individual tests can be located relatively.
        #
        # See also lldbtest.TestBase.setUpClass(cls).
        if len(testdirs) == 1 and os.path.basename(testdirs[0]) == 'test':
            os.environ["LLDB_TEST"] = os.path.join(rdir, 'test')
        else:
            os.environ["LLDB_TEST"] = rdir
    else:
        os.environ["LLDB_TEST"] = scriptPath
    pluginPath = os.path.join(scriptPath, 'plugins')

    # Append script dir and plugin dir to the sys.path.
    sys.path.append(scriptPath)
    sys.path.append(pluginPath)
    
    global ignore

    # The '-i' option is used to skip looking for lldb.py in the build tree.
    if ignore:
        return
        
    base = os.path.abspath(os.path.join(scriptPath, os.pardir))
    dbgPath = os.path.join(base, 'build', 'Debug', 'LLDB.framework',
                           'Resources', 'Python')
    relPath = os.path.join(base, 'build', 'Release', 'LLDB.framework',
                           'Resources', 'Python')
    baiPath = os.path.join(base, 'build', 'BuildAndIntegration',
                           'LLDB.framework', 'Resources', 'Python')

    lldbPath = None
    if os.path.isfile(os.path.join(dbgPath, 'lldb.py')):
        lldbPath = dbgPath
    elif os.path.isfile(os.path.join(relPath, 'lldb.py')):
        lldbPath = relPath
    elif os.path.isfile(os.path.join(baiPath, 'lldb.py')):
        lldbPath = baiPath

    if not lldbPath:
        print 'This script requires lldb.py to be in either ' + dbgPath + ',',
        print relPath + ', or ' + baiPath
        sys.exit(-1)

    # This is to locate the lldb.py module.  Insert it right after sys.path[0].
    sys.path[1:1] = [lldbPath]


def doDelay(delta):
    """Delaying startup for delta-seconds to facilitate debugger attachment."""
    def alarm_handler(*args):
        raise Exception("timeout")

    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(delta)
    sys.stdout.write("pid=%d\n" % os.getpid())
    sys.stdout.write("Enter RET to proceed (or timeout after %d seconds):" %
                     delta)
    sys.stdout.flush()
    try:
        text = sys.stdin.readline()
    except:
        text = ""
    signal.alarm(0)
    sys.stdout.write("proceeding...\n")
    pass


def visit(prefix, dir, names):
    """Visitor function for os.path.walk(path, visit, arg)."""

    global suite
    global regexp
    global filterspec
    global fs4all

    for name in names:
        if os.path.isdir(os.path.join(dir, name)):
            continue

        if '.py' == os.path.splitext(name)[1] and name.startswith(prefix):
            # Try to match the regexp pattern, if specified.
            if regexp:
                import re
                if re.search(regexp, name):
                    #print "Filename: '%s' matches pattern: '%s'" % (name, regexp)
                    pass
                else:
                    #print "Filename: '%s' does not match pattern: '%s'" % (name, regexp)
                    continue

            # We found a match for our test.  Add it to the suite.

            # Update the sys.path first.
            if not sys.path.count(dir):
                sys.path.insert(0, dir)
            base = os.path.splitext(name)[0]

            # Thoroughly check the filterspec against the base module and admit
            # the (base, filterspec) combination only when it makes sense.
            if filterspec:
                # Optimistically set the flag to True.
                filtered = True
                module = __import__(base)
                parts = filterspec.split('.')
                obj = module
                for part in parts:
                    try:
                        parent, obj = obj, getattr(obj, part)
                    except AttributeError:
                        # The filterspec has failed.
                        filtered = False
                        break
                # Forgo this module if the (base, filterspec) combo is invalid
                # and no '-g' option is specified
                if fs4all and not filtered:
                    continue
                
            # Add either the filtered test case or the entire test class.
            if filterspec and filtered:
                suite.addTests(
                    unittest2.defaultTestLoader.loadTestsFromName(filterspec, module))
            else:
                # A simple case of just the module name.  Also the failover case
                # from the filterspec branch when the (base, filterspec) combo
                # doesn't make sense.
                suite.addTests(unittest2.defaultTestLoader.loadTestsFromName(base))


def lldbLoggings():
    """Check and do lldb loggings if necessary."""

    # Turn on logging for debugging purposes if ${LLDB_LOG} environment variable is
    # defined.  Use ${LLDB_LOG} to specify the log file.
    ci = lldb.DBG.GetCommandInterpreter()
    res = lldb.SBCommandReturnObject()
    if ("LLDB_LOG" in os.environ):
        if ("LLDB_LOG_OPTION" in os.environ):
            lldb_log_option = os.environ["LLDB_LOG_OPTION"]
        else:
            lldb_log_option = "event process expr state api"
        ci.HandleCommand(
            "log enable -T -n -f " + os.environ["LLDB_LOG"] + " lldb " + lldb_log_option,
            res)
        if not res.Succeeded():
            raise Exception('log enable failed (check LLDB_LOG env variable.')
    # Ditto for gdb-remote logging if ${GDB_REMOTE_LOG} environment variable is defined.
    # Use ${GDB_REMOTE_LOG} to specify the log file.
    if ("GDB_REMOTE_LOG" in os.environ):
        if ("GDB_REMOTE_LOG_OPTION" in os.environ):
            gdb_remote_log_option = os.environ["GDB_REMOTE_LOG_OPTION"]
        else:
            gdb_remote_log_option = "packets process"
        ci.HandleCommand(
            "log enable -T -n -f " + os.environ["GDB_REMOTE_LOG"] + " process.gdb-remote "
            + gdb_remote_log_option,
            res)
        if not res.Succeeded():
            raise Exception('log enable failed (check GDB_REMOTE_LOG env variable.')

def getMyCommandLine():
    import subprocess
    ps = subprocess.Popen(['ps', '-o', "command=CMD", str(os.getpid())], stdout=subprocess.PIPE).communicate()[0]
    lines = ps.split('\n')
    cmd_line = lines[1]
    return cmd_line

# ======================================== #
#                                          #
# Execution of the test driver starts here #
#                                          #
# ======================================== #

#
# Start the actions by first parsing the options while setting up the test
# directories, followed by setting up the search paths for lldb utilities;
# then, we walk the directory trees and collect the tests into our test suite.
#
parseOptionsAndInitTestdirs()
setupSysPath()

#
# If '-d' is specified, do a delay of 10 seconds for the debugger to attach.
#
if delay:
    doDelay(10)

#
# If '-l' is specified, do not skip the long running tests.
if not skipLongRunningTest:
    os.environ["LLDB_SKIP_LONG_RUNNING_TEST"] = "NO"

#
# Walk through the testdirs while collecting tests.
#
for testdir in testdirs:
    os.path.walk(testdir, visit, 'Test')

#
# Now that we have loaded all the test cases, run the whole test suite.
#

# For the time being, let's bracket the test runner within the
# lldb.SBDebugger.Initialize()/Terminate() pair.
import lldb, atexit
# Update: the act of importing lldb now executes lldb.SBDebugger.Initialize(),
# there's no need to call it a second time.
#lldb.SBDebugger.Initialize()
atexit.register(lambda: lldb.SBDebugger.Terminate())

# Create a singleton SBDebugger in the lldb namespace.
lldb.DBG = lldb.SBDebugger.Create()

# Put the blacklist in the lldb namespace, to be used by lldb.TestBase.
lldb.blacklist = blacklist

# Put dont/just_do_python_api_test in the lldb namespace, too.
lldb.dont_do_python_api_test = dont_do_python_api_test
lldb.just_do_python_api_test = just_do_python_api_test

# Turn on lldb loggings if necessary.
lldbLoggings()

# Install the control-c handler.
unittest2.signals.installHandler()

# If sdir_name is not specified through the '-s sdir_name' option, get a
# timestamp string and export it as LLDB_SESSION_DIR environment var.  This will
# be used when/if we want to dump the session info of individual test cases
# later on.
#
# See also TestBase.dumpSessionInfo() in lldbtest.py.
if not sdir_name:
    import datetime
    # The windows platforms don't like ':' in the pathname.
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
    sdir_name = timestamp
os.environ["LLDB_SESSION_DIRNAME"] = sdir_name

sys.stderr.write("\nSession logs for test failures/errors will go into directory '%s'\n" % sdir_name)
sys.stderr.write("Command invoked: %s\n" % getMyCommandLine())

#
# Invoke the default TextTestRunner to run the test suite, possibly iterating
# over different configurations.
#

iterArchs = False
iterCompilers = False

from types import *
if "archs" in config:
    archs = config["archs"]
    if type(archs) is ListType and len(archs) >= 1:
        iterArchs = True
if "compilers" in config:
    compilers = config["compilers"]
    if type(compilers) is ListType and len(compilers) >= 1:
        iterCompilers = True

# Make a shallow copy of sys.path, we need to manipulate the search paths later.
# This is only necessary if we are relocated and with different configurations.
if rdir and (iterArchs or iterCompilers):
    old_sys_path = sys.path[:]
    old_stderr = sys.stderr
    old_stdout = sys.stdout
    new_stderr = None
    new_stdout = None

# Iterating over all possible architecture and compiler combinations.
for ia in range(len(archs) if iterArchs else 1):
    archConfig = ""
    if iterArchs:
        os.environ["ARCH"] = archs[ia]
        archConfig = "arch=%s" % archs[ia]
    for ic in range(len(compilers) if iterCompilers else 1):
        if iterCompilers:
            os.environ["CC"] = compilers[ic]
            configString = "%s compiler=%s" % (archConfig, compilers[ic])
        else:
            configString = archConfig

        if iterArchs or iterCompilers:
            # If we specified a relocated directory to run the test suite, do
            # the extra housekeeping to copy the testdirs to a configStringified
            # directory and to update sys.path before invoking the test runner.
            # The purpose is to separate the configuration-specific directories
            # from each other.
            if rdir:
                from string import maketrans
                from shutil import copytree, ignore_patterns

                # Translate ' ' to '-' for dir name.
                tbl = maketrans(' ', '-')
                configPostfix = configString.translate(tbl)
                newrdir = "%s.%s" % (rdir, configPostfix)

                # Copy the tree to a new directory with postfix name configPostfix.
                copytree(rdir, newrdir, ignore=ignore_patterns('*.pyc', '*.o', '*.d'))

                # Check whether we need to split stderr/stdout into configuration
                # specific files.
                if old_stderr.name != '<stderr>' and config.get('split_stderr'):
                    if new_stderr:
                        new_stderr.close()
                    new_stderr = open("%s.%s" % (old_stderr.name, configPostfix), "w")
                    sys.stderr = new_stderr
                if old_stdout.name != '<stdout>' and config.get('split_stdout'):
                    if new_stdout:
                        new_stdout.close()
                    new_stdout = open("%s.%s" % (old_stdout.name, configPostfix), "w")
                    sys.stdout = new_stdout

                # Update the LLDB_TEST environment variable to reflect new top
                # level test directory.
                #
                # See also lldbtest.TestBase.setUpClass(cls).
                if len(testdirs) == 1 and os.path.basename(testdirs[0]) == 'test':
                    os.environ["LLDB_TEST"] = os.path.join(newrdir, 'test')
                else:
                    os.environ["LLDB_TEST"] = newrdir

                # And update the Python search paths for modules.
                sys.path = [x.replace(rdir, newrdir, 1) for x in old_sys_path]

            # Output the configuration.
            sys.stderr.write("\nConfiguration: " + configString + "\n")

        #print "sys.stderr name is", sys.stderr.name
        #print "sys.stdout name is", sys.stdout.name

        # First, write out the number of collected test cases.
        sys.stderr.write(separator + "\n")
        sys.stderr.write("Collected %d test%s\n\n"
                         % (suite.countTestCases(),
                            suite.countTestCases() != 1 and "s" or ""))

        class LLDBTestResult(unittest2.TextTestResult):
            """
            Enforce a singleton pattern to allow introspection of test progress.

            Overwrite addError(), addFailure(), and addExpectedFailure() methods
            to enable each test instance to track its failure/error status.  It
            is used in the LLDB test framework to emit detailed trace messages
            to a log file for easier human inspection of test failres/errors.
            """
            __singleton__ = None
            __ignore_singleton__ = False

            def __init__(self, *args):
                if not LLDBTestResult.__ignore_singleton__ and LLDBTestResult.__singleton__:
                    raise Exception("LLDBTestResult instantiated more than once")
                super(LLDBTestResult, self).__init__(*args)
                LLDBTestResult.__singleton__ = self
                # Now put this singleton into the lldb module namespace.
                lldb.test_result = self
                # Computes the format string for displaying the counter.
                global suite
                counterWidth = len(str(suite.countTestCases()))
                self.fmt = "%" + str(counterWidth) + "d: "
                self.indentation = ' ' * (counterWidth + 2)
                # This counts from 1 .. suite.countTestCases().
                self.counter = 0

            def getDescription(self, test):
                doc_first_line = test.shortDescription()
                if self.descriptions and doc_first_line:
                    return '\n'.join((str(test), self.indentation + doc_first_line))
                else:
                    return str(test)

            def startTest(self, test):
                self.counter += 1
                if self.showAll:
                    self.stream.write(self.fmt % self.counter)
                super(LLDBTestResult, self).startTest(test)

            def addError(self, test, err):
                global sdir_has_content
                sdir_has_content = True
                super(LLDBTestResult, self).addError(test, err)
                method = getattr(test, "markError", None)
                if method:
                    method()

            def addFailure(self, test, err):
                global sdir_has_content
                sdir_has_content = True
                super(LLDBTestResult, self).addFailure(test, err)
                method = getattr(test, "markFailure", None)
                if method:
                    method()

            def addExpectedFailure(self, test, err):
                global sdir_has_content
                sdir_has_content = True
                super(LLDBTestResult, self).addExpectedFailure(test, err)
                method = getattr(test, "markExpectedFailure", None)
                if method:
                    method()

        # Invoke the test runner.
        if count == 1:
            result = unittest2.TextTestRunner(stream=sys.stderr,
                                              verbosity=verbose,
                                              failfast=failfast,
                                              resultclass=LLDBTestResult).run(suite)
        else:
            # We are invoking the same test suite more than once.  In this case,
            # mark __ignore_singleton__ flag as True so the signleton pattern is
            # not enforced.
            LLDBTestResult.__ignore_singleton__ = True
            for i in range(count):
                result = unittest2.TextTestRunner(stream=sys.stderr,
                                                  verbosity=verbose,
                                                  failfast=failfast,
                                                  resultclass=LLDBTestResult).run(suite)
        

if sdir_has_content:
    sys.stderr.write("Session logs for test failures/errors can be found in directory '%s'\n" % sdir_name)

# Terminate the test suite if ${LLDB_TESTSUITE_FORCE_FINISH} is defined.
# This should not be necessary now.
if ("LLDB_TESTSUITE_FORCE_FINISH" in os.environ):
    import subprocess
    print "Terminating Test suite..."
    subprocess.Popen(["/bin/sh", "-c", "kill %s; exit 0" % (os.getpid())])

# Exiting.
sys.exit(not result.wasSuccessful)
