"""
Test that you can set breakpoint commands successfully with the Python API's:
"""

import os
import re
import unittest2
import lldb, lldbutil
import sys
from lldbtest import *

class PythonBreakpointCommandSettingTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    my_var = 10

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test
    @dsym_test
    def test_step_out_with_dsym_python(self):
        """Test stepping out using avoid-no-debug with dsyms."""
        self.buildDsym()
        self.do_set_python_command_from_python()

    @python_api_test
    @dwarf_test
    def test_step_out_with_dwarf_python(self):
        """Test stepping out using avoid-no-debug with dsyms."""
        self.buildDwarf()
        self.do_set_python_command_from_python ()

    def setUp (self):
        TestBase.setUp(self)
        self.main_source = "main.c"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)


    def do_set_python_command_from_python (self):
        exe = os.path.join(os.getcwd(), "a.out")
        error = lldb.SBError()

        self.target = self.dbg.CreateTarget(exe)
        self.assertTrue(self.target, VALID_TARGET)

        body_bkpt = self.target.BreakpointCreateBySourceRegex("Set break point at this line.", self.main_source_spec)
        self.assertTrue(body_bkpt, VALID_BREAKPOINT)

        func_bkpt = self.target.BreakpointCreateBySourceRegex("Set break point at this line.", self.main_source_spec)
        self.assertTrue(func_bkpt, VALID_BREAKPOINT)

        PythonBreakpointCommandSettingTestCase.my_var = 10
        error = lldb.SBError()
        error = body_bkpt.SetScriptCallbackBody("\
import TestBreakpointCommandsFromPython\n\
TestBreakpointCommandsFromPython.PythonBreakpointCommandSettingTestCase.my_var = 20\n\
print 'Hit breakpoint'")
        self.assertTrue (error.Success(), "Failed to set the script callback body: %s."%(error.GetCString()))

        self.dbg.HandleCommand("command script import --allow-reload ./bktptcmd.py")
        func_bkpt.SetScriptCallbackFunction("bktptcmd.function")

        # We will use the function that touches a text file, so remove it first:
        self.RemoveTempFile("output2.txt")

        # Now launch the process, and do not stop at entry point.
        self.process = self.target.LaunchSimple (None, None, self.get_process_working_directory())

        self.assertTrue(self.process, PROCESS_IS_VALID)

        # Now finish, and make sure the return value is correct.
        threads = lldbutil.get_threads_stopped_at_breakpoint (self.process, body_bkpt)
        self.assertTrue(len(threads) == 1, "Stopped at inner breakpoint.")
        self.thread = threads[0]
    
        self.assertTrue(PythonBreakpointCommandSettingTestCase.my_var == 20)

        # Check for the function version as well, which produced this file:
        # Remember to clean up after ourselves...
        self.assertTrue(os.path.isfile("output2.txt"),
                        "'output2.txt' exists due to breakpoint command for breakpoint function.")
        self.RemoveTempFile("output2.txt")

        
if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
