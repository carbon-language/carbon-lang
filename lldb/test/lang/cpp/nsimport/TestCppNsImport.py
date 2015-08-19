"""
Tests imported namespaces in C++.
"""
import lldb
from lldbtest import *
import lldbutil

class TestCppNsImport(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipUnlessDarwin
    @dsym_test
    def test_with_dsym_and_run_command(self):
        """Tests imported namespaces in C++."""
        self.buildDsym()
        self.check()

    @dwarf_test
    def test_with_dwarf_and_run_command(self):
        """Tests imported namespaces in C++."""
        self.buildDwarf()
        self.check()

    def setUp(self):
        TestBase.setUp(self)

    def check(self):
        """Tests imported namespaces in C++."""

        # Get main source file
        src_file = "main.cpp"
        src_file_spec = lldb.SBFileSpec(src_file)
        self.assertTrue(src_file_spec.IsValid(), "Main source file")

        # Get the path of the executable
        cwd = os.getcwd()
        exe_file = "a.out"
        exe_path  = os.path.join(cwd, exe_file)

        # Load the executable
        target = self.dbg.CreateTarget(exe_path)
        self.assertTrue(target.IsValid(), VALID_TARGET)

        # Break on main function
        break_0 = target.BreakpointCreateBySourceRegex("// break 0", src_file_spec)
        self.assertTrue(break_0.IsValid() and break_0.GetNumLocations() >= 1, VALID_BREAKPOINT)

        # Launch the process
        args = None
        env = None
        process = target.LaunchSimple(args, env, self.get_process_working_directory())
        self.assertTrue(process.IsValid(), PROCESS_IS_VALID)

        # Get the thread of the process
        self.assertTrue(process.GetState() == lldb.eStateStopped, PROCESS_STOPPED)
        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint)

        # Get current fream of the thread at the breakpoint
        frame = thread.GetSelectedFrame()

        # Test imported namespaces
        test_result = frame.EvaluateExpression("n")
        self.assertTrue(test_result.IsValid() and test_result.GetValueAsSigned() == 1, "n = 1")

        test_result = frame.EvaluateExpression("N::n")
        self.assertTrue(test_result.IsValid() and test_result.GetValueAsSigned() == 1, "N::n = 1")

        test_result = frame.EvaluateExpression("nested")
        self.assertTrue(test_result.IsValid() and test_result.GetValueAsSigned() == 3, "nested = 3")

        test_result = frame.EvaluateExpression("anon")
        self.assertTrue(test_result.IsValid() and test_result.GetValueAsSigned() == 2, "anon = 2")


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
