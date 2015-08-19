"""
Test that global operators are found and evaluated.
"""
import lldb
from lldbtest import *
import lldbutil

class TestCppGlobalOperators(TestBase):
    
    mydir = TestBase.compute_mydir(__file__)
    
    @skipUnlessDarwin
    @dsym_test
    def test_with_dsym_and_run_command(self):
        self.buildDsym()
        self.check()

    @dwarf_test
    def test_with_dwarf_and_run_command(self):
        self.buildDwarf()
        self.check()

    def setUp(self):
        TestBase.setUp(self)

    def check(self):
        # Get main source file
        src_file = "main.cpp"
        src_file_spec = lldb.SBFileSpec(src_file)
        self.assertTrue(src_file_spec.IsValid(), "Main source file")
        
        # Get the path of the executable
        cwd = self.get_process_working_directory()
        exe_file = "a.out"
        exe_path  = os.path.join(cwd, exe_file)
        
        # Load the executable
        target = self.dbg.CreateTarget(exe_path)
        self.assertTrue(target.IsValid(), VALID_TARGET)

        # Break on main function
        main_breakpoint = target.BreakpointCreateBySourceRegex("// break here", src_file_spec)
        self.assertTrue(main_breakpoint.IsValid() and main_breakpoint.GetNumLocations() >= 1, VALID_BREAKPOINT)

        # Launch the process
        args = None
        env = None
        process = target.LaunchSimple(args, env, cwd)
        self.assertTrue(process.IsValid(), PROCESS_IS_VALID)

        # Get the thread of the process
        self.assertTrue(process.GetState() == lldb.eStateStopped, PROCESS_STOPPED)
        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint)

        # Check if global operators are evaluated 
        frame = thread.GetSelectedFrame()

        test_result = frame.EvaluateExpression("operator==(s1, s2)")
        self.assertTrue(test_result.IsValid() and test_result.GetValue() == "false", "operator==(s1, s2) = false")
 
        test_result = frame.EvaluateExpression("operator==(s1, s3)")
        self.assertTrue(test_result.IsValid() and test_result.GetValue() == "true", "operator==(s1, s3) = true")

        test_result = frame.EvaluateExpression("operator==(s2, s3)")
        self.assertTrue(test_result.IsValid() and test_result.GetValue() == "false", "operator==(s2, s3) = false")


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
