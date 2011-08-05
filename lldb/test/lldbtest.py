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
$ export PYTHONPATH=/Volumes/data/lldb/svn/trunk/build/Debug/LLDB.framework/Resources/Python:$LLDB_TEST:$LLDB_TEST/plugins:$LLDB_TEST/pexpect-2.4
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
# LLDB_COMMAND_TRACE and LLDB_DO_CLEANUP are set from '-t' and '-r dir' options.

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

PROCESS_EXITED = "Process exited successfully"

PROCESS_STOPPED = "Process status should be stopped"

RUN_SUCCEEDED = "Process is launched successfully"

RUN_COMPLETED = "Process exited successfully"

BACKTRACE_DISPLAYED_CORRECTLY = "Backtrace displayed correctly"

BREAKPOINT_CREATED = "Breakpoint created successfully"

BREAKPOINT_STATE_CORRECT = "Breakpoint state is correct"

BREAKPOINT_PENDING_CREATED = "Pending breakpoint created successfully"

BREAKPOINT_HIT_ONCE = "Breakpoint resolved with hit cout = 1"

BREAKPOINT_HIT_TWICE = "Breakpoint resolved with hit cout = 2"

BREAKPOINT_HIT_THRICE = "Breakpoint resolved with hit cout = 3"

OBJECT_PRINTED_CORRECTLY = "Object printed correctly"

SOURCE_DISPLAYED_CORRECTLY = "Source code displayed correctly"

STEP_OUT_SUCCEEDED = "Thread step-out succeeded"

STOPPED_DUE_TO_EXC_BAD_ACCESS = "Process should be stopped due to bad access exception"

STOPPED_DUE_TO_BREAKPOINT = "Process should be stopped due to breakpoint"

STOPPED_DUE_TO_BREAKPOINT_WITH_STOP_REASON_AS = "%s, %s" % (
    STOPPED_DUE_TO_BREAKPOINT, "instead, the actual stop reason is: '%s'")

STOPPED_DUE_TO_BREAKPOINT_CONDITION = "Stopped due to breakpoint condition"

STOPPED_DUE_TO_BREAKPOINT_IGNORE_COUNT = "Stopped due to breakpoint and ignore count"

STOPPED_DUE_TO_SIGNAL = "Process state is stopped due to signal"

STOPPED_DUE_TO_STEP_IN = "Process state is stopped due to step in"

DATA_TYPES_DISPLAYED_CORRECTLY = "Data type(s) displayed correctly"

VALID_BREAKPOINT = "Got a valid breakpoint"

VALID_BREAKPOINT_LOCATION = "Got a valid breakpoint location"

VALID_COMMAND_INTERPRETER = "Got a valid command interpreter"

VALID_FILESPEC = "Got a valid filespec"

VALID_MODULE = "Got a valid module"

VALID_PROCESS = "Got a valid process"

VALID_SYMBOL = "Got a valid symbol"

VALID_TARGET = "Got a valid target"

VALID_VARIABLE = "Got a valid variable"

VARIABLES_DISPLAYED_CORRECTLY = "Variable(s) displayed correctly"


def CMD_MSG(str):
    '''A generic "Command '%s' returns successfully" message generator.'''
    return "Command '%s' returns successfully" % str

def EXP_MSG(str, exe):
    '''A generic "'%s' returns expected result" message generator if exe.
    Otherwise, it generates "'%s' matches expected result" message.'''
    return "'%s' %s expected result" % (str, 'returns' if exe else 'matches')

def SETTING_MSG(setting):
    '''A generic "Value of setting '%s' is correct" message generator.'''
    return "Value of setting '%s' is correct" % setting

def EnvArray():
    """Returns an env variable array from the os.environ map object."""
    return map(lambda k,v: k+"="+v, os.environ.keys(), os.environ.values())

def line_number(filename, string_to_match):
    """Helper function to return the line number of the first matched string."""
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            if line.find(string_to_match) != -1:
                # Found our match.
                return i+1
    raise Exception("Unable to find '%s' within file %s" % (string_to_match, filename))

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
# Return a tuple (stdoutdata, stderrdata).
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
    process = Popen(stdout=PIPE, stderr=PIPE, *popenargs, **kwargs)
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
    return (output, error)

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

def builder_module():
    return __import__("builder_" + sys.platform)

#
# Decorators for categorizing test cases.
#

from functools import wraps
def python_api_test(func):
    """Decorate the item as a Python API only test."""
    if isinstance(func, type) and issubclass(func, unittest2.TestCase):
        raise Exception("@python_api_test can only be used to decorate a test method")
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            if lldb.dont_do_python_api_test:
                self.skipTest("python api tests")
        except AttributeError:
            pass
        return func(self, *args, **kwargs)

    # Mark this function as such to separate them from lldb command line tests.
    wrapper.__python_api_test__ = True
    return wrapper

from functools import wraps
def benchmarks_test(func):
    """Decorate the item as a benchmarks test."""
    if isinstance(func, type) and issubclass(func, unittest2.TestCase):
        raise Exception("@benchmarks_test can only be used to decorate a test method")
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            if not lldb.just_do_benchmarks_test:
                self.skipTest("benchmarks tests")
        except AttributeError:
            pass
        return func(self, *args, **kwargs)

    # Mark this function as such to separate them from the regular tests.
    wrapper.__benchmarks_test__ = True
    return wrapper

class Base(unittest2.TestCase):
    """
    Abstract base for performing lldb (see TestBase) or other generic tests (see
    BenchBase for one example).  lldbtest.Base works with the test driver to
    accomplish things.
    
    """
    # The concrete subclass should override this attribute.
    mydir = None

    # Keep track of the old current working directory.
    oldcwd = None

    def TraceOn(self):
        """Returns True if we are in trace mode (tracing detailed test execution)."""
        return traceAlways

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
            module = builder_module()
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

    def setUp(self):
        """Fixture for unittest test case setup.

        It works with the test driver to conditionally skip tests and does other
        initializations."""
        #import traceback
        #traceback.print_stack()

        if "LLDB_EXEC" in os.environ:
            self.lldbExec = os.environ["LLDB_EXEC"]

        # Assign the test method name to self.testMethodName.
        #
        # For an example of the use of this attribute, look at test/types dir.
        # There are a bunch of test cases under test/types and we don't want the
        # module cacheing subsystem to be confused with executable name "a.out"
        # used for all the test cases.
        self.testMethodName = self._testMethodName

        # Python API only test is decorated with @python_api_test,
        # which also sets the "__python_api_test__" attribute of the
        # function object to True.
        try:
            if lldb.just_do_python_api_test:
                testMethod = getattr(self, self._testMethodName)
                if getattr(testMethod, "__python_api_test__", False):
                    pass
                else:
                    self.skipTest("non python api test")
        except AttributeError:
            pass

        # Benchmarks test is decorated with @benchmarks_test,
        # which also sets the "__benchmarks_test__" attribute of the
        # function object to True.
        try:
            if lldb.just_do_benchmarks_test:
                testMethod = getattr(self, self._testMethodName)
                if getattr(testMethod, "__benchmarks_test__", False):
                    pass
                else:
                    self.skipTest("non benchmarks test")
        except AttributeError:
            pass

        # This is for the case of directly spawning 'lldb'/'gdb' and interacting
        # with it using pexpect.
        self.child = None
        self.child_prompt = "(lldb) "
        # If the child is interacting with the embedded script interpreter,
        # there are two exits required during tear down, first to quit the
        # embedded script interpreter and second to quit the lldb command
        # interpreter.
        self.child_in_script_interpreter = False

        # These are for customized teardown cleanup.
        self.dict = None
        self.doTearDownCleanup = False
        # And in rare cases where there are multiple teardown cleanups.
        self.dicts = []
        self.doTearDownCleanups = False

        # Create a string buffer to record the session info, to be dumped into a
        # test case specific file if test failure is encountered.
        self.session = StringIO.StringIO()

        # Optimistically set __errored__, __failed__, __expected__ to False
        # initially.  If the test errored/failed, the session info
        # (self.session) is then dumped into a session specific file for
        # diagnosis.
        self.__errored__    = False
        self.__failed__     = False
        self.__expected__   = False
        # We are also interested in unexpected success.
        self.__unexpected__ = False

        # See addTearDownHook(self, hook) which allows the client to add a hook
        # function to be run during tearDown() time.
        self.hooks = []

        # See HideStdout(self).
        self.sys_stdout_hidden = False

    def HideStdout(self):
        """Hide output to stdout from the user.

        During test execution, there might be cases where we don't want to show the
        standard output to the user.  For example,

            self.runCmd(r'''sc print "\n\n\tHello!\n"''')

        tests whether command abbreviation for 'script' works or not.  There is no
        need to show the 'Hello' output to the user as long as the 'script' command
        succeeds and we are not in TraceOn() mode (see the '-t' option).

        In this case, the test method calls self.HideStdout(self) to redirect the
        sys.stdout to a null device, and restores the sys.stdout upon teardown.

        Note that you should only call this method at most once during a test case
        execution.  Any subsequent call has no effect at all."""
        if self.sys_stdout_hidden:
            return

        self.sys_stdout_hidden = True
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        def restore_stdout():
            sys.stdout = old_stdout
        self.addTearDownHook(restore_stdout)

    # =======================================================================
    # Methods for customized teardown cleanups as well as execution of hooks.
    # =======================================================================

    def setTearDownCleanup(self, dictionary=None):
        """Register a cleanup action at tearDown() time with a dictinary"""
        self.dict = dictionary
        self.doTearDownCleanup = True

    def addTearDownCleanup(self, dictionary):
        """Add a cleanup action at tearDown() time with a dictinary"""
        self.dicts.append(dictionary)
        self.doTearDownCleanups = True

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
        """Fixture for unittest test case teardown."""
        #import traceback
        #traceback.print_stack()

        # This is for the case of directly spawning 'lldb' and interacting with it
        # using pexpect.
        import pexpect
        if self.child and self.child.isalive():
            with recording(self, traceAlways) as sbuf:
                print >> sbuf, "tearing down the child process...."
            if self.child_in_script_interpreter:
                self.child.sendline('quit()')
                self.child.expect_exact(self.child_prompt)
            self.child.sendline('quit')
            try:
                self.child.expect(pexpect.EOF)
            except:
                pass

        # Check and run any hook functions.
        for hook in reversed(self.hooks):
            with recording(self, traceAlways) as sbuf:
                print >> sbuf, "Executing tearDown hook:", getsource_if_available(hook)
            hook()

        del self.hooks

        # Perform registered teardown cleanup.
        if doCleanup and self.doTearDownCleanup:
            module = builder_module()
            if not module.cleanup(self, dictionary=self.dict):
                raise Exception("Don't know how to do cleanup with dictionary: " + self.dict)

        # In rare cases where there are multiple teardown cleanups added.
        if doCleanup and self.doTearDownCleanups:
            module = builder_module()
            if self.dicts:
                for dict in reversed(self.dicts):
                    if not module.cleanup(self, dictionary=dict):
                        raise Exception("Don't know how to do cleanup with dictionary: " + dict)

        # Decide whether to dump the session info.
        self.dumpSessionInfo()

    # =========================================================
    # Various callbacks to allow introspection of test progress
    # =========================================================

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

    def markUnexpectedSuccess(self):
        """Callback invoked when an unexpected success occurred."""
        self.__unexpected__ = True
        with recording(self, False) as sbuf:
            # False because there's no need to write "unexpected success" to the
            # stderr twice.
            # Once by the Python unittest framework, and a second time by us.
            print >> sbuf, "unexpected success"

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
        elif self.__unexpected__:
            prefix = "UnexpectedSuccess"
        else:
            # Simply return, there's no session info to dump!
            return

        if not self.__unexpected__:
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
            print >> f, "./dotest.py %s -v -t -f %s.%s" % (self.getRunOptions(),
                                                           self.__class__.__name__,
                                                           self._testMethodName)

    # ====================================================
    # Config. methods supported through a plugin interface
    # (enables reading of the current test configuration)
    # ====================================================

    def getArchitecture(self):
        """Returns the architecture in effect the test suite is running with."""
        module = builder_module()
        return module.getArchitecture()

    def getCompiler(self):
        """Returns the compiler in effect the test suite is running with."""
        module = builder_module()
        return module.getCompiler()

    def getRunOptions(self):
        """Command line option for -A and -C to run this test again, called from
        self.dumpSessionInfo()."""
        arch = self.getArchitecture()
        comp = self.getCompiler()
        if not arch and not comp:
            return ""
        else:
            return "%s %s" % ("-A "+arch if arch else "",
                              "-C "+comp if comp else "")

    # ==================================================
    # Build methods supported through a plugin interface
    # ==================================================

    def buildDefault(self, architecture=None, compiler=None, dictionary=None):
        """Platform specific way to build the default binaries."""
        module = builder_module()
        if not module.buildDefault(self, architecture, compiler, dictionary):
            raise Exception("Don't know how to build default binary")

    def buildDsym(self, architecture=None, compiler=None, dictionary=None):
        """Platform specific way to build binaries with dsym info."""
        module = builder_module()
        if not module.buildDsym(self, architecture, compiler, dictionary):
            raise Exception("Don't know how to build binary with dsym")

    def buildDwarf(self, architecture=None, compiler=None, dictionary=None):
        """Platform specific way to build binaries with dwarf maps."""
        module = builder_module()
        if not module.buildDwarf(self, architecture, compiler, dictionary):
            raise Exception("Don't know how to build binary with dwarf")


class TestBase(Base):
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
              - populate the test method name
              - create/get a debugger set with synchronous mode (self.dbg)
              - get the command interpreter from with the debugger (self.ci)
              - create a result object for use with the command interpreter
                (self.res)
              - plus other stuffs

        - The tearDown method tries to perform some necessary cleanup on behalf
          of the test to return the debugger to a good state for the next test.
          These include:
              - execute any tearDown hooks registered by the test method with
                TestBase.addTearDownHook(); examples can be found in
                settings/TestSettings.py
              - kill the inferior process associated with each target, if any,
                and, then delete the target from the debugger's target list
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

    # Maximum allowed attempts when launching the inferior process.
    # Can be overridden by the LLDB_MAX_LAUNCH_COUNT environment variable.
    maxLaunchCount = 3;

    # Time to wait before the next launching attempt in second(s).
    # Can be overridden by the LLDB_TIME_WAIT_NEXT_LAUNCH environment variable.
    timeWaitNextLaunch = 1.0;

    def doDelay(self):
        """See option -w of dotest.py."""
        if ("LLDB_WAIT_BETWEEN_TEST_CASES" in os.environ and
            os.environ["LLDB_WAIT_BETWEEN_TEST_CASES"] == 'YES'):
            waitTime = 1.0
            if "LLDB_TIME_WAIT_BETWEEN_TEST_CASES" in os.environ:
                waitTime = float(os.environ["LLDB_TIME_WAIT_BETWEEN_TEST_CASES"])
            time.sleep(waitTime)

    def setUp(self):
        #import traceback
        #traceback.print_stack()

        # Works with the test driver to conditionally skip tests via decorators.
        Base.setUp(self)

        try:
            if lldb.blacklist:
                className = self.__class__.__name__
                classAndMethodName = "%s.%s" % (className, self._testMethodName)
                if className in lldb.blacklist:
                    self.skipTest(lldb.blacklist.get(className))
                elif classAndMethodName in lldb.blacklist:
                    self.skipTest(lldb.blacklist.get(classAndMethodName))
        except AttributeError:
            pass

        # Insert some delay between successive test cases if specified.
        self.doDelay()

        if "LLDB_MAX_LAUNCH_COUNT" in os.environ:
            self.maxLaunchCount = int(os.environ["LLDB_MAX_LAUNCH_COUNT"])

        if "LLDB_TIME_WAIT_NEXT_LAUNCH" in os.environ:
            self.timeWaitNextLaunch = float(os.environ["LLDB_TIME_WAIT_NEXT_LAUNCH"])

        # Create the debugger instance if necessary.
        try:
            self.dbg = lldb.DBG
        except AttributeError:
            self.dbg = lldb.SBDebugger.Create()

        if not self.dbg:
            raise Exception('Invalid debugger instance')

        # We want our debugger to be synchronous.
        self.dbg.SetAsync(False)

        # Retrieve the associated command interpreter instance.
        self.ci = self.dbg.GetCommandInterpreter()
        if not self.ci:
            raise Exception('Could not get the command interpreter')

        # And the result object.
        self.res = lldb.SBCommandReturnObject()

    def tearDown(self):
        #import traceback
        #traceback.print_stack()

        Base.tearDown(self)

        # Delete the target(s) from the debugger as a general cleanup step.
        # This includes terminating the process for each target, if any.
        # We'd like to reuse the debugger for our next test without incurring
        # the initialization overhead.
        targets = []
        for target in self.dbg:
            if target:
                targets.append(target)
                process = target.GetProcess()
                if process:
                    rc = self.invoke(process, "Kill")
                    self.assertTrue(rc.Success(), PROCESS_KILLED)
        for target in targets:
            self.dbg.DeleteTarget(target)

        del self.dbg

    def runCmd(self, cmd, msg=None, check=True, trace=False):
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

            if self.res.Succeeded():
                break
            elif running:
                # For process launch, wait some time before possible next try.
                time.sleep(self.timeWaitNextLaunch)
                with recording(self, True) as sbuf:
                    print >> sbuf, "Command '" + cmd + "' failed!"

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

    # =================================================
    # Misc. helper methods for debugging test execution
    # =================================================

    def DebugSBValue(self, val):
        """Debug print a SBValue object, if traceAlways is True."""
        from lldbutil import value_type_to_str

        if not traceAlways:
            return

        err = sys.stderr
        err.write(val.GetName() + ":\n")
        err.write('\t' + "TypeName      -> " + val.GetTypeName()            + '\n')
        err.write('\t' + "ByteSize      -> " + str(val.GetByteSize())       + '\n')
        err.write('\t' + "NumChildren   -> " + str(val.GetNumChildren())    + '\n')
        err.write('\t' + "Value         -> " + str(val.GetValue())          + '\n')
        err.write('\t' + "ValueType     -> " + value_type_to_str(val.GetValueType()) + '\n')
        err.write('\t' + "Summary       -> " + str(val.GetSummary())        + '\n')
        err.write('\t' + "IsPointerType -> " + str(val.TypeIsPointerType()) + '\n')
        err.write('\t' + "Location      -> " + val.GetLocation()            + '\n')

    def DebugSBType(self, type):
        """Debug print a SBType object, if traceAlways is True."""
        if not traceAlways:
            return

        err = sys.stderr
        err.write(type.GetName() + ":\n")
        err.write('\t' + "ByteSize        -> " + str(type.GetByteSize())     + '\n')
        err.write('\t' + "IsPointerType   -> " + str(type.IsPointerType())   + '\n')
        err.write('\t' + "IsReferenceType -> " + str(type.IsReferenceType()) + '\n')

    def DebugPExpect(self, child):
        """Debug the spwaned pexpect object."""
        if not traceAlways:
            return

        print child
