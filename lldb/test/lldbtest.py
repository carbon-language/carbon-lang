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
$ export PYTHONPATH=/Volumes/data/lldb/svn/trunk/build/Debug/LLDB.framework/Resources/Python:$LLDB_TEST
$ echo $LLDB_TEST
/Volumes/data/lldb/svn/trunk/test
$ echo $PYTHONPATH
/Volumes/data/lldb/svn/trunk/build/Debug/LLDB.framework/Resources/Python:/Volumes/data/lldb/svn/trunk/test
$ python function_types/TestFunctionTypes.py
.
----------------------------------------------------------------------
Ran 1 test in 0.363s

OK
$ 
"""

import os
import unittest2
import lldb
import traceback

class TestBase(unittest2.TestCase):
    """This LLDB abstract base class is meant to be subclassed."""

    # The concrete subclass should override this attribute.
    mydir = None

    def setUp(self):
        #traceback.print_stack()

        # Fail fast if 'mydir' attribute is not overridden.
        if not self.mydir or len(self.mydir) == 0:
            raise Exception("Subclasses must override the 'mydir' attribute.")
        # Save old working directory.
        self.oldcwd = os.getcwd()

        # Change current working directory if ${LLDB_TEST} is defined.
        # See also dotest.py which sets up ${LLDB_TEST}.
        if ("LLDB_TEST" in os.environ):
            os.chdir(os.path.join(os.environ["LLDB_TEST"], self.mydir));

        # Create the debugger instance if necessary.
        try:
            self.dbg = lldb.DBG
        except AttributeError:
            self.dbg = lldb.SBDebugger.Create()

        if not self.dbg.IsValid():
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
        del self.dbg

        # Restore old working directory.
        os.chdir(self.oldcwd)
