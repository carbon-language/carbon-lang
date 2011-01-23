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
        self.assertTrue(target.IsValid(), VALID_TARGET)

        # Now create a breakpoint on main.c by name 'c'.
        breakpoint = target.BreakpointCreateByName('c', 'a.out')
        #print "breakpoint:", breakpoint
        self.assertTrue(breakpoint.IsValid() and
                        breakpoint.GetNumLocations() == 1,
                        VALID_BREAKPOINT)

        # Now launch the process, and do not stop at the entry point.
        # Note that we don't assign the process to self.process as in other test
        # cases.  We want the inferior to run till it exits and there's no need
        # for the testing framework to kill the inferior upon tearDown().
        error = lldb.SBError()
        process = target.Launch (None, None, os.ctermid(), os.ctermid(), os.ctermid(), None, 0, False, error)

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
                print "frame:", frame
                #print "frame.FindValue('val', lldb.eValueTypeVariableArgument)", frame.FindValue('val', lldb.eValueTypeVariableArgument).GetValue(frame)
                #print "frame.FindValue('ch', lldb.eValueTypeVariableArgument)", frame.FindValue('ch', lldb.eValueTypeVariableArgument).GetValue(frame)
                #print "frame.EvaluateExpression('val'):", frame.EvaluateExpression('val').GetValue(frame)
                #print "frame.EvaluateExpression('ch'):", frame.EvaluateExpression('ch').GetValue(frame)
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
                from lldbutil import lldb_iter
                for val in lldb_iter(valList, 'GetSize', 'GetValueAtIndex'):
                    #self.DebugSBValue(frame, val)
                    argList.append("(%s)%s=%s" % (val.GetTypeName(),
                                                  val.GetName(),
                                                  val.GetValue(frame)))
                print >> session, "%s(%s)" % (name, ", ".join(argList))

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
