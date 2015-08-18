import lldb
from lldbtest import *
import lldbutil

class TestCppChainedCalls(TestBase):
    
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
        cwd = os.getcwd() 
        exe_file = "a.out"
        exe_path  = os.path.join(cwd, exe_file)
        
        # Load the executable
        target = self.dbg.CreateTarget(exe_path)
        self.assertTrue(target.IsValid(), VALID_TARGET)

        # Break on main function
        main_breakpoint = target.BreakpointCreateBySourceRegex("Break here", src_file_spec)
        self.assertTrue(main_breakpoint.IsValid() and main_breakpoint.GetNumLocations() >= 1, VALID_BREAKPOINT)

        # Launch the process
        args = None
        env = None
        process = target.LaunchSimple(args, env, cwd)
        self.assertTrue(process.IsValid(), PROCESS_IS_VALID)

        # Get the thread of the process
        self.assertTrue(process.GetState() == lldb.eStateStopped, PROCESS_STOPPED)
        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint)

        # Get frame for current thread 
        frame = thread.GetSelectedFrame()
        
        # Test chained calls
        test_result = frame.EvaluateExpression("g(f(12345))")
        self.assertTrue(test_result.IsValid() and test_result.GetValueAsSigned() == 12345, "g(f(12345)) = 12345")

        test_result = frame.EvaluateExpression("q(p()).a")
        self.assertTrue(test_result.IsValid() and test_result.GetValueAsSigned() == 12345678, "q(p()).a = 12345678")

        test_result = frame.EvaluateExpression("(p() + r()).a")
        self.assertTrue(test_result.IsValid() and test_result.GetValueAsSigned() == 22345678, "(p() + r()).a = 22345678")

        test_result = frame.EvaluateExpression("q(p() + r()).a")
        self.assertTrue(test_result.IsValid() and test_result.GetValueAsSigned() == 22345678, "q(p() + r()).a = 22345678")

        test_result = frame.EvaluateExpression("g(f(6700) + f(89))")
        self.assertTrue(test_result.IsValid() and test_result.GetValueAsSigned() == 6789, "g(f(6700) + f(89)) = 6789")

        test_result = frame.EvaluateExpression("g(f(g(f(300) + f(40))) + f(5))")
        self.assertTrue(test_result.IsValid() and test_result.GetValueAsSigned() == 345, "g(f(g(f(300) + f(40))) + f(5)) = 345")

        test_result = frame.EvaluateExpression("getb(makeb(), 789)")
        self.assertTrue(test_result.IsValid() and test_result.GetValueAsSigned() == 789, "getb(makeb(), 789) = 789")

        test_result = frame.EvaluateExpression("(*c).a")
        self.assertTrue(test_result.IsValid() and test_result.GetValueAsSigned() == 5678, "(*c).a = 5678")

        test_result = frame.EvaluateExpression("(*c + *c).a")
        self.assertTrue(test_result.IsValid() and test_result.GetValueAsSigned() == 11356, "(*c + *c).a = 11356")

        test_result = frame.EvaluateExpression("q(*c + *c).a")
        self.assertTrue(test_result.IsValid() and test_result.GetValueAsSigned() == 11356, "q(*c + *c).a = 11356")

        test_result = frame.EvaluateExpression("make_int().get_type()")
        self.assertTrue(test_result.IsValid() and test_result.GetValue() == "INT", "make_int().get_type() = \"INT\"")


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
