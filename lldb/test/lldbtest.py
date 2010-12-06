"""
LLDB module which provides the abstract base class of lldb test case.

The concrete subclass can override lldbtest.TesBase in order to inherit the
common behavior for unitest.TestCase.setUp/tearDown implemented in this file.

The subclass should override the attribute mydir in order for the python runtime
to locate the individual test cases when running as part of a large test suite
or when running each test case as a separate python invocation.

./dotest.py provides a test driver which sets up the environment to run the
entire test suite.  Users who want to run a test case on its own can specify the
LLDB_TEST and PYTHONPATH environment variables, for example:

$ export LLDB_TEST=$PWD
$ export PYTHONPATH=/Volumes/data/lldb/svn/trunk/build/Debug/LLDB.framework/Resources/Python:$LLDB_TEST:$LLDB_TEST/plugins
$ echo $LLDB_TEST
/Volumes/data/lldb/svn/trunk/test
$ echo $PYTHONPATH
/Volumes/data/lldb/svn/trunk/build/Debug/LLDB.framework/Resources/Python:/Volumes/data/lldb/svn/trunk/test:/Volumes/data/lldb/svn/trunk/test/plugins
$ python function_types/TestFunctionTypes.py
.
----------------------------------------------------------------------
Ran 1 test in 0.363s

OK
$ LLDB_COMMAND_TRACE=YES python array_types/TestArrayTypes.py

...

runCmd: breakpoint set -f main.c -l 42
output: Breakpoint created: 1: file ='main.c', line = 42, locations = 1

runCmd: run
output: Launching '/Volumes/data/lldb/svn/trunk/test/array_types/a.out'  (x86_64)

...

runCmd: frame variable strings
output: (char *[4]) strings = {
  (char *) strings[0] = 0x0000000100000f0c "Hello",
  (char *) strings[1] = 0x0000000100000f12 "Hola",
  (char *) strings[2] = 0x0000000100000f17 "Bonjour",
  (char *) strings[3] = 0x0000000100000f1f "Guten Tag"
}

runCmd: frame variable char_16
output: (char [16]) char_16 = {
  (char) char_16[0] = 'H',
  (char) char_16[1] = 'e',
  (char) char_16[2] = 'l',
  (char) char_16[3] = 'l',
  (char) char_16[4] = 'o',
  (char) char_16[5] = ' ',
  (char) char_16[6] = 'W',
  (char) char_16[7] = 'o',
  (char) char_16[8] = 'r',
  (char) char_16[9] = 'l',
  (char) char_16[10] = 'd',
  (char) char_16[11] = '\n',
  (char) char_16[12] = '\0',
  (char) char_16[13] = '\0',
  (char) char_16[14] = '\0',
  (char) char_16[15] = '\0'
}

runCmd: frame variable ushort_matrix
output: (unsigned short [2][3]) ushort_matrix = {
  (unsigned short [3]) ushort_matrix[0] = {
    (unsigned short) ushort_matrix[0][0] = 0x0001,
    (unsigned short) ushort_matrix[0][1] = 0x0002,
    (unsigned short) ushort_matrix[0][2] = 0x0003
  },
  (unsigned short [3]) ushort_matrix[1] = {
    (unsigned short) ushort_matrix[1][0] = 0x000b,
    (unsigned short) ushort_matrix[1][1] = 0x0016,
    (unsigned short) ushort_matrix[1][2] = 0x0021
  }
}

runCmd: frame variable long_6
output: (long [6]) long_6 = {
  (long) long_6[0] = 1,
  (long) long_6[1] = 2,
  (long) long_6[2] = 3,
  (long) long_6[3] = 4,
  (long) long_6[4] = 5,
  (long) long_6[5] = 6
}

.
----------------------------------------------------------------------
Ran 1 test in 0.349s

OK
$ 
"""

import os, sys, traceback
import re
from subprocess import *
import StringIO
import time
import types
import unittest2
import lldb

# See also dotest.parseOptionsAndInitTestdirs(), where the environment variables
# LLDB_COMMAND_TRACE and LLDB_NO_CLEANUP are set from '-t' and '-r dir' options.

# By default, traceAlways is False.
if "LLDB_COMMAND_TRACE" in os.environ and os.environ["LLDB_COMMAND_TRACE"]=="YES":
    traceAlways = True
else:
    traceAlways = False

# By default, doCleanup is True.
if "LLDB_DO_CLEANUP" in os.environ and os.environ["LLDB_DO_CLEANUP"]=="NO":
    doCleanup = False
else:
    doCleanup = True


#
# Some commonly used assert messages.
#

COMMAND_FAILED_AS_EXPECTED = "Command has failed as expected"

CURRENT_EXECUTABLE_SET = "Current executable set successfully"

PROCESS_IS_VALID = "Process is valid"

PROCESS_KILLED = "Process is killed successfully"

RUN_SUCCEEDED = "Process is launched successfully"

RUN_COMPLETED = "Process exited successfully"

BACKTRACE_DISPLAYED_CORRECTLY = "Backtrace displayed correctly"

BREAKPOINT_CREATED = "Breakpoint created successfully"

BREAKPOINT_STATE_CORRECT = "Breakpoint state is correct"

BREAKPOINT_PENDING_CREATED = "Pending breakpoint created successfully"

BREAKPOINT_HIT_ONCE = "Breakpoint resolved with hit cout = 1"

BREAKPOINT_HIT_TWICE = "Breakpoint resolved with hit cout = 2"

BREAKPOINT_HIT_THRICE = "Breakpoint resolved with hit cout = 3"

STEP_OUT_SUCCEEDED = "Thread step-out succeeded"

PROCESS_STOPPED = "Process status should be stopped"

STOPPED_DUE_TO_BREAKPOINT = "Process should be stopped due to breakpoint"

STOPPED_DUE_TO_BREAKPOINT_WITH_STOP_REASON_AS = "%s, %s" % (
    STOPPED_DUE_TO_BREAKPOINT, "instead, the actual stop reason is: '%s'")

STOPPED_DUE_TO_BREAKPOINT_CONDITION = "Stopped due to breakpoint condition"

STOPPED_DUE_TO_SIGNAL = "Process state is stopped due to signal"

STOPPED_DUE_TO_STEP_IN = "Process state is stopped due to step in"

DATA_TYPES_DISPLAYED_CORRECTLY = "Data type(s) displayed correctly"

VALID_BREAKPOINT = "Got a valid breakpoint"

VALID_BREAKPOINT_LOCATION = "Got a valid breakpoint location"

VALID_FILESPEC = "Got a valid filespec"

VALID_PROCESS = "Got a valid process"

VALID_TARGET = "Got a valid target"

VARIABLES_DISPLAYED_CORRECTLY = "Variable(s) displayed correctly"


#
# And a generic "Command '%s' returns successfully" message generator.
#
def CMD_MSG(str):
    return "Command '%s' returns successfully" % str

#
# And a generic "'%s' returns expected result" message generator if exe.
# Otherwise, it's "'%s' matches expected result"
#
def EXP_MSG(str, exe):
    return "'%s' %s expected result" % (str, 'returns' if exe else 'matches')

#
# And a generic "Value of setting '%s' is correct" message generator.
#
def SETTING_MSG(setting):
    return "Value of setting '%s' is correct" % setting

#
# Returns an env variable array from the os.environ map object.
#
def EnvArray():
    return map(lambda k,v: k+"="+v, os.environ.keys(), os.environ.values())

def line_number(filename, string_to_match):
    """Helper function to return the line number of the first matched string."""
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            if line.find(string_to_match) != -1:
                # Found our match.
                return i+1
    raise Exception("Unable to find %s within file %s" % (string_to_match, filename))

def pointer_size():
    """Return the pointer size of the host system."""
    import ctypes
    a_pointer = ctypes.c_void_p(0xffff)
    return 8 * ctypes.sizeof(a_pointer)


class recording(StringIO.StringIO):
    """
    A nice little context manager for recording the debugger interactions into
    our session object.  If trace flag is ON, it also emits the interactions
    into the stderr.
    """
    def __init__(self, test, trace):
        """Create a StringIO instance; record the session obj and trace flag."""
        StringIO.StringIO.__init__(self)
        self.session = test.session if test else None
        self.trace = trace

    def __enter__(self):
        """
        Context management protocol on entry to the body of the with statement.
        Just return the StringIO object.
        """
        return self

    def __exit__(self, type, value, tb):
        """
        Context management protocol on exit from the body of the with statement.
        If trace is ON, it emits the recordings into stderr.  Always add the
        recordings to our session object.  And close the StringIO object, too.
        """
        if self.trace:
            print >> sys.stderr, self.getvalue()
        if self.session:
            print >> self.session, self.getvalue()
        self.close()

# From 2.7's subprocess.check_output() convenience function.
def system(*popenargs, **kwargs):
    r"""Run command with arguments and return its output as a byte string.

    If the exit code was non-zero it raises a CalledProcessError.  The
    CalledProcessError object will have the return code in the returncode
    attribute and output in the output attribute.

    The arguments are the same as for the Popen constructor.  Example:

    >>> check_output(["ls", "-l", "/dev/null"])
    'crw-rw-rw- 1 root root 1, 3 Oct 18  2007 /dev/null\n'

    The stdout argument is not allowed as it is used internally.
    To capture standard error in the result, use stderr=STDOUT.

    >>> check_output(["/bin/sh", "-c",
    ...               "ls -l non_existent_file ; exit 0"],
    ...              stderr=STDOUT)
    'ls: non_existent_file: No such file or directory\n'
    """

    # Assign the sender object to variable 'test' and remove it from kwargs.
    test = kwargs.pop('sender', None)

    if 'stdout' in kwargs:
        raise ValueError('stdout argument not allowed, it will be overridden.')
    process = Popen(stdout=PIPE, *popenargs, **kwargs)
    output, error = process.communicate()
    retcode = process.poll()

    with recording(test, traceAlways) as sbuf:
        if isinstance(popenargs, types.StringTypes):
            args = [popenargs]
        else:
            args = list(popenargs)
        print >> sbuf
        print >> sbuf, "os command:", args
        print >> sbuf, "stdout:", output
        print >> sbuf, "stderr:", error
        print >> sbuf, "retcode:", retcode
        print >> sbuf

    if retcode:
        cmd = kwargs.get("args")
        if cmd is None:
            cmd = popenargs[0]
        raise CalledProcessError(retcode, cmd)
    return output

def getsource_if_available(obj):
    """
    Return the text of the source code for an object if available.  Otherwise,
    a print representation is returned.
    """
    import inspect
    try:
        return inspect.getsource(obj)
    except:
        return repr(obj)

class TestBase(unittest2.TestCase):
    """
    This abstract base class is meant to be subclassed.  It provides default
    implementations for setUpClass(), tearDownClass(), setUp(), and tearDown(),
    among other things.

    Important things for test class writers:

        - Overwrite the mydir class attribute, otherwise your test class won't
          run.  It specifies the relative directory to the top level 'test' so
          the test harness can change to the correct working directory before
          running your test.

        - The setUp method sets up things to facilitate subsequent interactions
          with the debugger as part of the test.  These include:
              - create/get a debugger set with synchronous mode (self.dbg)
              - get the command interpreter from with the debugger (self.ci)
              - create a result object for use with the command interpreter
                (self.result)
              - plus other stuffs

        - The tearDown method tries to perform some necessary cleanup on behalf
          of the test to return the debugger to a good state for the next test.
          These include:
              - execute any tearDown hooks registered by the test method with
                TestBase.addTearDownHook(); examples can be found in
                settings/TestSettings.py
              - kill the inferior process launched during the test method
                    - if by 'run' or 'process launch' command, 'process kill'
                      command is used
                    - if the test method uses LLDB Python API to launch process,
                      it should assign the process object to self.process; that
                      way, tearDown will use self.process.Kill() on the object
              - perform build cleanup before running the next test method in the
                same test class; examples of registering for this service can be
                found in types/TestIntegerTypes.py with the call:
                    - self.setTearDownCleanup(dictionary=d)

        - Similarly setUpClass and tearDownClass perform classwise setup and
          teardown fixtures.  The tearDownClass method invokes a default build
          cleanup for the entire test class;  also, subclasses can implement the
          classmethod classCleanup(cls) to perform special class cleanup action.

        - The instance methods runCmd and expect are used heavily by existing
          test cases to send a command to the command interpreter and to perform
          string/pattern matching on the output of such command execution.  The
          expect method also provides a mode to peform string/pattern matching
          without running a command.

        - The build methods buildDefault, buildDsym, and buildDwarf are used to
          build the binaries used during a particular test scenario.  A plugin
          should be provided for the sys.platform running the test suite.  The
          Mac OS X implementation is located in plugins/darwin.py.

    """

    @classmethod
    def skipLongRunningTest(cls):
        """
        By default, we skip long running test case.
        This can be overridden by passing '-l' to the test driver (dotest.py).
        """
        if "LLDB_SKIP_LONG_RUNNING_TEST" in os.environ and "NO" == os.environ["LLDB_SKIP_LONG_RUNNING_TEST"]:
            return False
        else:
            return True

    # The concrete subclass should override this attribute.
    mydir = None

    # State pertaining to the inferior process, if any.
    # This reflects inferior process started through the command interface with
    # either the lldb "run" or "process launch" command.
    # See also self.runCmd().
    runStarted = False

    # Maximum allowed attempts when launching the inferior process.
    # Can be overridden by the LLDB_MAX_LAUNCH_COUNT environment variable.
    maxLaunchCount = 3;

    # Time to wait before the next launching attempt in second(s).
    # Can be overridden by the LLDB_TIME_WAIT_NEXT_LAUNCH environment variable.
    timeWaitNextLaunch = 1.0;

    # Keep track of the old current working directory.
    oldcwd = None

    @classmethod
    def setUpClass(cls):
        """
        Python unittest framework class setup fixture.
        Do current directory manipulation.
        """

        # Fail fast if 'mydir' attribute is not overridden.
        if not cls.mydir or len(cls.mydir) == 0:
            raise Exception("Subclasses must override the 'mydir' attribute.")
        # Save old working directory.
        cls.oldcwd = os.getcwd()

        # Change current working directory if ${LLDB_TEST} is defined.
        # See also dotest.py which sets up ${LLDB_TEST}.
        if ("LLDB_TEST" in os.environ):
            if traceAlways:
                print >> sys.stderr, "Change dir to:", os.path.join(os.environ["LLDB_TEST"], cls.mydir)
            os.chdir(os.path.join(os.environ["LLDB_TEST"], cls.mydir))

    @classmethod
    def tearDownClass(cls):
        """
        Python unittest framework class teardown fixture.
        Do class-wide cleanup.
        """

        if doCleanup:
            # First, let's do the platform-specific cleanup.
            module = __import__(sys.platform)
            if not module.cleanup():
                raise Exception("Don't know how to do cleanup")

            # Subclass might have specific cleanup function defined.
            if getattr(cls, "classCleanup", None):
                if traceAlways:
                    print >> sys.stderr, "Call class-specific cleanup function for class:", cls
                try:
                    cls.classCleanup()
                except:
                    exc_type, exc_value, exc_tb = sys.exc_info()
                    traceback.print_exception(exc_type, exc_value, exc_tb)

        # Restore old working directory.
        if traceAlways:
            print >> sys.stderr, "Restore dir to:", cls.oldcwd
        os.chdir(cls.oldcwd)

    def setUp(self):
        #import traceback
        #traceback.print_stack()

        if lldb.blacklist:
            className = self.__class__.__name__
            classAndMethodName = "%s.%s" % (className, self._testMethodName)
            if className in lldb.blacklist:
                self.skipTest(lldb.blacklist.get(className))
            elif classAndMethodName in lldb.blacklist:
                self.skipTest(lldb.blacklist.get(classAndMethodName))

        if ("LLDB_WAIT_BETWEEN_TEST_CASES" in os.environ and
            os.environ["LLDB_WAIT_BETWEEN_TEST_CASES"] == 'YES'):
            waitTime = 1.0
            if "LLDB_TIME_WAIT_BETWEEN_TEST_CASES" in os.environ:
                waitTime = float(os.environ["LLDB_TIME_WAIT_BETWEEN_TEST_CASES"])
            time.sleep(waitTime)

        if "LLDB_MAX_LAUNCH_COUNT" in os.environ:
            self.maxLaunchCount = int(os.environ["LLDB_MAX_LAUNCH_COUNT"])

        if "LLDB_TIME_WAIT_NEXT_LAUNCH" in os.environ:
            self.timeWaitNextLaunch = float(os.environ["LLDB_TIME_WAIT_NEXT_LAUNCH"])

        # Create the debugger instance if necessary.
        try:
            self.dbg = lldb.DBG
        except AttributeError:
            self.dbg = lldb.SBDebugger.Create()

        if not self.dbg.IsValid():
            raise Exception('Invalid debugger instance')

        # We want our debugger to be synchronous.
        self.dbg.SetAsync(False)

        # There is no process associated with the debugger as yet.
        # See also self.tearDown() where it checks whether self.process has a
        # valid reference and calls self.process.Kill() to kill the process.
        self.process = None

        # Retrieve the associated command interpreter instance.
        self.ci = self.dbg.GetCommandInterpreter()
        if not self.ci:
            raise Exception('Could not get the command interpreter')

        # And the result object.
        self.res = lldb.SBCommandReturnObject()

        # These are for customized teardown cleanup.
        self.dict = None
        self.doTearDownCleanup = False

        # Create a string buffer to record the session info, to be dumped into a
        # test case specific file if test failure is encountered.
        self.session = StringIO.StringIO()

        # Optimistically set __errored__, __failed__, __expected__ to False
        # initially.  If the test errored/failed, the session info
        # (self.session) is then dumped into a session specific file for
        # diagnosis.
        self.__errored__ = False
        self.__failed__ = False
        self.__expected__ = False

        # See addTearDownHook(self, hook) which allows the client to add a hook
        # function to be run during tearDown() time.
        self.hooks = []

    def markError(self):
        """Callback invoked when an error (unexpected exception) errored."""
        self.__errored__ = True
        with recording(self, False) as sbuf:
            # False because there's no need to write "ERROR" to the stderr twice.
            # Once by the Python unittest framework, and a second time by us.
            print >> sbuf, "ERROR"

    def markFailure(self):
        """Callback invoked when a failure (test assertion failure) occurred."""
        self.__failed__ = True
        with recording(self, False) as sbuf:
            # False because there's no need to write "FAIL" to the stderr twice.
            # Once by the Python unittest framework, and a second time by us.
            print >> sbuf, "FAIL"

    def markExpectedFailure(self):
        """Callback invoked when an expected failure/error occurred."""
        self.__expected__ = True
        with recording(self, False) as sbuf:
            # False because there's no need to write "expected failure" to the
            # stderr twice.
            # Once by the Python unittest framework, and a second time by us.
            print >> sbuf, "expected failure"

    def dumpSessionInfo(self):
        """
        Dump the debugger interactions leading to a test error/failure.  This
        allows for more convenient postmortem analysis.

        See also LLDBTestResult (dotest.py) which is a singlton class derived
        from TextTestResult and overwrites addError, addFailure, and
        addExpectedFailure methods to allow us to to mark the test instance as
        such.
        """

        # We are here because self.tearDown() detected that this test instance
        # either errored or failed.  The lldb.test_result singleton contains
        # two lists (erros and failures) which get populated by the unittest
        # framework.  Look over there for stack trace information.
        #
        # The lists contain 2-tuples of TestCase instances and strings holding
        # formatted tracebacks.
        #
        # See http://docs.python.org/library/unittest.html#unittest.TestResult.
        if self.__errored__:
            pairs = lldb.test_result.errors
            prefix = 'Error'
        elif self.__failed__:
            pairs = lldb.test_result.failures
            prefix = 'Failure'
        elif self.__expected__:
            pairs = lldb.test_result.expectedFailures
            prefix = 'ExpectedFailure'
        else:
            # Simply return, there's no session info to dump!
            return

        for test, traceback in pairs:
            if test is self:
                print >> self.session, traceback

        dname = os.path.join(os.environ["LLDB_TEST"],
                             os.environ["LLDB_SESSION_DIRNAME"])
        if not os.path.isdir(dname):
            os.mkdir(dname)
        fname = os.path.join(dname, "%s-%s.log" % (prefix, self.id()))
        with open(fname, "w") as f:
            import datetime
            print >> f, "Session info generated @", datetime.datetime.now().ctime()
            print >> f, self.session.getvalue()
            print >> f, "To rerun this test, issue the following command from the 'test' directory:\n"
            print >> f, "%s ./dotest.py -v -t -f %s.%s" % (self.getRunSpec(),
                                                           self.__class__.__name__,
                                                           self._testMethodName)

    def setTearDownCleanup(self, dictionary=None):
        """Register a cleanup action at tearDown() time with a dictinary"""
        self.dict = dictionary
        self.doTearDownCleanup = True

    def addTearDownHook(self, hook):
        """
        Add a function to be run during tearDown() time.

        Hooks are executed in a first come first serve manner.
        """
        if callable(hook):
            with recording(self, traceAlways) as sbuf:
                print >> sbuf, "Adding tearDown hook:", getsource_if_available(hook)
            self.hooks.append(hook)

    def tearDown(self):
        #import traceback
        #traceback.print_stack()

        # Check and run any hook functions.
        for hook in self.hooks:
            with recording(self, traceAlways) as sbuf:
                print >> sbuf, "Executing tearDown hook:", getsource_if_available(hook)
            hook()

        # Terminate the current process being debugged, if any.
        if self.runStarted:
            self.runCmd("process kill", PROCESS_KILLED, check=False)
        elif self.process and self.process.IsValid():
            rc = self.invoke(self.process, "Kill")
            self.assertTrue(rc.Success(), PROCESS_KILLED)
            del self.process

        del self.dbg
        del self.hooks

        # Perform registered teardown cleanup.
        if doCleanup and self.doTearDownCleanup:
            module = __import__(sys.platform)
            if not module.cleanup(self, dictionary=self.dict):
                raise Exception("Don't know how to do cleanup")

        # Decide whether to dump the session info.
        self.dumpSessionInfo()

    def runCmd(self, cmd, msg=None, check=True, trace=False, setCookie=True):
        """
        Ask the command interpreter to handle the command and then check its
        return status.
        """
        # Fail fast if 'cmd' is not meaningful.
        if not cmd or len(cmd) == 0:
            raise Exception("Bad 'cmd' parameter encountered")

        trace = (True if traceAlways else trace)

        running = (cmd.startswith("run") or cmd.startswith("process launch"))

        for i in range(self.maxLaunchCount if running else 1):
            self.ci.HandleCommand(cmd, self.res)

            with recording(self, trace) as sbuf:
                print >> sbuf, "runCmd:", cmd
                if not check:
                    print >> sbuf, "check of return status not required"
                if self.res.Succeeded():
                    print >> sbuf, "output:", self.res.GetOutput()
                else:
                    print >> sbuf, "runCmd failed!"
                    print >> sbuf, self.res.GetError()

            if running:
                # For process launch, wait some time before possible next try.
                time.sleep(self.timeWaitNextLaunch)

            if self.res.Succeeded():
                break
            elif running:
                with recording(self, True) as sbuf:
                    print >> sbuf, "Command '" + cmd + "' failed!"

        # Modify runStarted only if "run" or "process launch" was encountered.
        if running:
            self.runStarted = running and setCookie

        if check:
            self.assertTrue(self.res.Succeeded(),
                            msg if msg else CMD_MSG(cmd))

    def expect(self, str, msg=None, patterns=None, startstr=None, substrs=None, trace=False, error=False, matching=True, exe=True):
        """
        Similar to runCmd; with additional expect style output matching ability.

        Ask the command interpreter to handle the command and then check its
        return status.  The 'msg' parameter specifies an informational assert
        message.  We expect the output from running the command to start with
        'startstr', matches the substrings contained in 'substrs', and regexp
        matches the patterns contained in 'patterns'.

        If the keyword argument error is set to True, it signifies that the API
        client is expecting the command to fail.  In this case, the error stream
        from running the command is retrieved and compared against the golden
        input, instead.

        If the keyword argument matching is set to False, it signifies that the API
        client is expecting the output of the command not to match the golden
        input.

        Finally, the required argument 'str' represents the lldb command to be
        sent to the command interpreter.  In case the keyword argument 'exe' is
        set to False, the 'str' is treated as a string to be matched/not-matched
        against the golden input.
        """
        trace = (True if traceAlways else trace)

        if exe:
            # First run the command.  If we are expecting error, set check=False.
            # Pass the assert message along since it provides more semantic info.
            self.runCmd(str, msg=msg, trace = (True if trace else False), check = not error)

            # Then compare the output against expected strings.
            output = self.res.GetError() if error else self.res.GetOutput()

            # If error is True, the API client expects the command to fail!
            if error:
                self.assertFalse(self.res.Succeeded(),
                                 "Command '" + str + "' is expected to fail!")
        else:
            # No execution required, just compare str against the golden input.
            output = str
            with recording(self, trace) as sbuf:
                print >> sbuf, "looking at:", output

        # The heading says either "Expecting" or "Not expecting".
        heading = "Expecting" if matching else "Not expecting"

        # Start from the startstr, if specified.
        # If there's no startstr, set the initial state appropriately.
        matched = output.startswith(startstr) if startstr else (True if matching else False)

        if startstr:
            with recording(self, trace) as sbuf:
                print >> sbuf, "%s start string: %s" % (heading, startstr)
                print >> sbuf, "Matched" if matched else "Not matched"

        # Look for sub strings, if specified.
        keepgoing = matched if matching else not matched
        if substrs and keepgoing:
            for str in substrs:
                matched = output.find(str) != -1
                with recording(self, trace) as sbuf:
                    print >> sbuf, "%s sub string: %s" % (heading, str)
                    print >> sbuf, "Matched" if matched else "Not matched"
                keepgoing = matched if matching else not matched
                if not keepgoing:
                    break

        # Search for regular expression patterns, if specified.
        keepgoing = matched if matching else not matched
        if patterns and keepgoing:
            for pattern in patterns:
                # Match Objects always have a boolean value of True.
                matched = bool(re.search(pattern, output))
                with recording(self, trace) as sbuf:
                    print >> sbuf, "%s pattern: %s" % (heading, pattern)
                    print >> sbuf, "Matched" if matched else "Not matched"
                keepgoing = matched if matching else not matched
                if not keepgoing:
                    break

        self.assertTrue(matched if matching else not matched,
                        msg if msg else EXP_MSG(str, exe))

    def invoke(self, obj, name, trace=False):
        """Use reflection to call a method dynamically with no argument."""
        trace = (True if traceAlways else trace)
        
        method = getattr(obj, name)
        import inspect
        self.assertTrue(inspect.ismethod(method),
                        name + "is a method name of object: " + str(obj))
        result = method()
        with recording(self, trace) as sbuf:
            print >> sbuf, str(method) + ":",  result
        return result

    def breakAfterLaunch(self, process, func, trace=False):
        """
        Perform some dancees after LaunchProcess() to break at func name.

        Return True if we can successfully break at the func name in due time.
        """
        trace = (True if traceAlways else trace)

        count = 0
        while True:
            # The stop reason of the thread should be breakpoint.
            thread = process.GetThreadAtIndex(0)
            SR = thread.GetStopReason()
            with recording(self, trace) as sbuf:
                print >> sbuf, "StopReason =", StopReasonString(SR)

            if SR == StopReasonEnum("Breakpoint"):
                frame = thread.GetFrameAtIndex(0)
                name = frame.GetFunction().GetName()
                with recording(self, trace) as sbuf:
                    print >> sbuf, "function =", name
                if (name == func):
                    # We got what we want; now break out of the loop.
                    return True

            # The inferior is in a transient state; continue the process.
            time.sleep(1.0)
            with recording(self, trace) as sbuf:
                print >> sbuf, "Continuing the process:", process
            process.Continue()

            count = count + 1
            if count == 15:
                with recording(self, trace) as sbuf:
                    print >> sbuf, "Reached 15 iterations, giving up..."
                # Enough iterations already, break out of the loop.
                return False

            # End of while loop.


    def getCompiler(self):
        """Returns the compiler in effect the test suite is now running with."""
        module = __import__(sys.platform)
        return module.getCompiler()

    def getRunSpec(self):
        """Environment variable spec to run this test again, invoked from within
        dumpSessionInfo()."""
        module = __import__(sys.platform)
        return module.getRunSpec()

    def buildDefault(self, architecture=None, compiler=None, dictionary=None):
        """Platform specific way to build the default binaries."""
        module = __import__(sys.platform)
        if not module.buildDefault(self, architecture, compiler, dictionary):
            raise Exception("Don't know how to build default binary")

    def buildDsym(self, architecture=None, compiler=None, dictionary=None):
        """Platform specific way to build binaries with dsym info."""
        module = __import__(sys.platform)
        if not module.buildDsym(self, architecture, compiler, dictionary):
            raise Exception("Don't know how to build binary with dsym")

    def buildDwarf(self, architecture=None, compiler=None, dictionary=None):
        """Platform specific way to build binaries with dwarf maps."""
        module = __import__(sys.platform)
        if not module.buildDwarf(self, architecture, compiler, dictionary):
            raise Exception("Don't know how to build binary with dwarf")

    def DebugSBValue(self, frame, val):
        """Debug print a SBValue object, if traceAlways is True."""
        from lldbutil import ValueTypeString

        if not traceAlways:
            return

        err = sys.stderr
        err.write(val.GetName() + ":\n")
        err.write('\t' + "TypeName      -> " + val.GetTypeName()            + '\n')
        err.write('\t' + "ByteSize      -> " + str(val.GetByteSize())       + '\n')
        err.write('\t' + "NumChildren   -> " + str(val.GetNumChildren())    + '\n')
        err.write('\t' + "Value         -> " + str(val.GetValue(frame))     + '\n')
        err.write('\t' + "ValueType     -> " + ValueTypeString(val.GetValueType()) + '\n')
        err.write('\t' + "Summary       -> " + str(val.GetSummary(frame))   + '\n')
        err.write('\t' + "IsPointerType -> " + str(val.TypeIsPointerType()) + '\n')
        err.write('\t' + "Location      -> " + val.GetLocation(frame)       + '\n')

