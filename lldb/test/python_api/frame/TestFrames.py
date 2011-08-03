"""
Use lldb Python SBFrame API to get the argument values of the call stacks.
"""

import os, time
import re
import unittest2
import lldb, lldbutil
from lldbtest import *

class FrameAPITestCase(TestBase):

    mydir = os.path.join("python_api", "frame")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test
    def test_get_arg_vals_for_call_stack_with_dsym(self):
        """Exercise SBFrame.GetVariables() API to get argument vals."""
        self.buildDsym()
        self.do_get_arg_vals()

    @python_api_test
    def test_get_arg_vals_for_call_stack_with_dwarf(self):
        """Exercise SBFrame.GetVariables() API to get argument vals."""
        self.buildDwarf()
        self.do_get_arg_vals()

    def do_get_arg_vals(self):
        """Get argument vals for the call stack when stopped on a breakpoint."""
        exe = os.path.join(os.getcwd(), "a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Now create a breakpoint on main.c by name 'c'.
        breakpoint = target.BreakpointCreateByName('c', 'a.out')
        #print "breakpoint:", breakpoint
        self.assertTrue(breakpoint and
                        breakpoint.GetNumLocations() == 1,
                        VALID_BREAKPOINT)

        # Now launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(None, None, os.getcwd())

        process = target.GetProcess()
        self.assertTrue(process.GetState() == lldb.eStateStopped,
                        PROCESS_STOPPED)

        # Keeps track of the number of times 'a' is called where it is within a
        # depth of 3 of the 'c' leaf function.
        callsOfA = 0

        import StringIO
        session = StringIO.StringIO()
        while process.GetState() == lldb.eStateStopped:
            thread = process.GetThreadAtIndex(0)
            # Inspect at most 3 frames.
            numFrames = min(3, thread.GetNumFrames())
            for i in range(numFrames):
                frame = thread.GetFrameAtIndex(i)
                if self.TraceOn():
                    print "frame:", frame

                name = frame.GetFunction().GetName()
                if name == 'a':
                    callsOfA = callsOfA + 1

                # We'll inspect only the arguments for the current frame:
                #
                # arguments     => True
                # locals        => False
                # statics       => False
                # in_scope_only => True
                valList = frame.GetVariables(True, False, False, True)
                argList = []
                for val in valList:
                    argList.append("(%s)%s=%s" % (val.GetTypeName(),
                                                  val.GetName(),
                                                  val.GetValue()))
                print >> session, "%s(%s)" % (name, ", ".join(argList))
                
                # Also check the generic pc & stack pointer.  We can't test their absolute values,
                # but they should be valid.  Uses get_GPRs() from the lldbutil module.
                gpr_reg_set = lldbutil.get_GPRs(frame)
                pc_value = gpr_reg_set.GetChildMemberWithName("pc")
                self.assertTrue (pc_value, "We should have a valid PC.")
                self.assertTrue (int(pc_value.GetValue(), 0) == frame.GetPC(), "PC gotten as a value should equal frame's GetPC")
                sp_value = gpr_reg_set.GetChildMemberWithName("sp")
                self.assertTrue (sp_value, "We should have a valid Stack Pointer.")
                self.assertTrue (int(sp_value.GetValue(), 0) == frame.GetSP(), "SP gotten as a value should equal frame's GetSP")

            print >> session, "---"
            process.Continue()

        # At this point, the inferior process should have exited.
        self.assertTrue(process.GetState() == lldb.eStateExited, PROCESS_EXITED)

        # Expect to find 'a' on the call stacks two times.
        self.assertTrue(callsOfA == 2,
                        "Expect to find 'a' on the call stacks two times")
        # By design, the 'a' call frame has the following arg vals:
        #     o a((int)val=1, (char)ch='A')
        #     o a((int)val=3, (char)ch='A')
        if self.TraceOn():
            print "Full stack traces when stopped on the breakpoint 'c':"
            print session.getvalue()
        self.expect(session.getvalue(), "Argugment values displayed correctly",
                    exe=False,
            substrs = ["a((int)val=1, (char)ch='A')",
                       "a((int)val=3, (char)ch='A')"])


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
