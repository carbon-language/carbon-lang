"""Test calling methods on super."""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestObjCSuperMethod(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line numbers to break inside main().
        self.main_source = "class.m"
        self.break_line = line_number(
            self.main_source, '// Set breakpoint here.')

    @add_test_categories(['pyapi'])
    def test_with_python_api(self):
        """Test calling methods on super."""
        self.build()
        exe = self.getBuildArtifact("a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        bpt = target.BreakpointCreateByLocation(
            self.main_source, self.break_line)
        self.assertTrue(bpt, VALID_BREAKPOINT)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())

        self.assertTrue(process, PROCESS_IS_VALID)

        # The stop reason of the thread should be breakpoint.
        thread_list = lldbutil.get_threads_stopped_at_breakpoint(process, bpt)

        # Make sure we stopped at the first breakpoint.
        self.assertTrue(
            len(thread_list) != 0,
            "No thread stopped at our breakpoint.")
        self.assertEquals(len(thread_list), 1,
                        "More than one thread stopped at our breakpoint.")

        # Now make sure we can call a function in the class method we've
        # stopped in.
        frame = thread_list[0].GetFrameAtIndex(0)
        self.assertTrue(frame, "Got a valid frame 0 frame.")

        cmd_value = frame.EvaluateExpression("[self get]")
        self.assertTrue(cmd_value.IsValid())
        self.assertEquals(cmd_value.GetValueAsUnsigned(), 2)

        cmd_value = frame.EvaluateExpression("[super get]")
        self.assertTrue(cmd_value.IsValid())
        self.assertEquals(cmd_value.GetValueAsUnsigned(), 1)
