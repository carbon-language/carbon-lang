"""
Test getting return-values correctly when stepping out
"""

import os, time
import re
import unittest2
import lldb, lldbutil
from lldbtest import *

class ReturnValueTestCase(TestBase):

    mydir = os.path.join("functionalities", "return-value")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test
    def test_with_dsym_python(self):
        """Test getting return values from stepping out with dsyms."""
        self.buildDsym()
        self.do_return_value()

    @python_api_test
    def test_with_dwarf_python(self):
        """Test getting return values from stepping out."""
        self.buildDwarf()
        self.do_return_value()

    def do_return_value(self):
        """Test getting return values from stepping out."""
        exe = os.path.join(os.getcwd(), "a.out")
        error = lldb.SBError()

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        inner_sint_bkpt = target.BreakpointCreateByName("inner_sint", exe)
        self.assertTrue(inner_sint_bkpt, VALID_BREAKPOINT)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(None, None, os.getcwd())

        self.assertTrue(process, PROCESS_IS_VALID)

        # The stop reason of the thread should be breakpoint.
        self.assertTrue(process.GetState() == lldb.eStateStopped,
                        STOPPED_DUE_TO_BREAKPOINT)

        # Now finish, and make sure the return value is correct.
        thread = lldbutil.get_stopped_thread (process, lldb.eStopReasonBreakpoint)

        thread.StepOut();

        self.assertTrue (process.GetState() == lldb.eStateStopped)
        self.assertTrue (thread.GetStopReason() == lldb.eStopReasonPlanComplete)

        frame = thread.GetFrameAtIndex(0)
        fun_name = frame.GetFunctionName()
        self.assertTrue (fun_name == "outer_sint")

        return_value = thread.GetStopReturnValue()
        self.assertTrue (return_value.IsValid())

        in_int = frame.FindVariable ("value").GetValueAsSigned(error)
        self.assertTrue (error.Success())
        ret_int = return_value.GetValueAsSigned(error)
        self.assertTrue (error.Success())
        self.assertTrue (in_int == ret_int)

        # Run again and we will stop in inner_sint the second time outer_sint is called.  
        #Then test stepping out two frames at once:

        process.Continue()
        thread_list = lldbutil.get_threads_stopped_at_breakpoint (process, inner_sint_bkpt)
        self.assertTrue(len(thread_list) == 1)
        thread = thread_list[0]

        frame = thread.GetFrameAtIndex(1)
        fun_name = frame.GetFunctionName ()
        self.assertTrue (fun_name == "outer_sint")
        in_int = frame.FindVariable ("value").GetValueAsSigned(error)
        self.assertTrue (error.Success())

        thread.StepOutOfFrame (frame)

        self.assertTrue (process.GetState() == lldb.eStateStopped)
        self.assertTrue (thread.GetStopReason() == lldb.eStopReasonPlanComplete)
        frame = thread.GetFrameAtIndex(0)
        fun_name = frame.GetFunctionName()
        self.assertTrue (fun_name == "main")

        ret_value = thread.GetStopReturnValue()
        self.assertTrue (return_value.IsValid())
        ret_int = ret_value.GetValueAsSigned (error)
        self.assertTrue (error.Success())
        self.assertTrue (in_int == ret_int)
        
        # Now try some simple returns that have different types:
        inner_float_bkpt = target.BreakpointCreateByName("inner_float", exe)
        self.assertTrue(inner_float_bkpt, VALID_BREAKPOINT)
        process.Continue()
        thread_list = lldbutil.get_threads_stopped_at_breakpoint (process, inner_float_bkpt)
        self.assertTrue (len(thread_list) == 1)
        thread = thread_list[0]

        frame = thread.GetFrameAtIndex(0)
        in_value = frame.FindVariable ("value")
        in_float = float (in_value.GetValue())
        thread.StepOut()

        self.assertTrue (process.GetState() == lldb.eStateStopped)
        self.assertTrue (thread.GetStopReason() == lldb.eStopReasonPlanComplete)

        frame = thread.GetFrameAtIndex(0)
        fun_name = frame.GetFunctionName()
        self.assertTrue (fun_name == "outer_float")

        return_value = thread.GetStopReturnValue()
        self.assertTrue (return_value.IsValid())
        return_float = float (return_value.GetValue())

        self.assertTrue(in_float == return_float)

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
