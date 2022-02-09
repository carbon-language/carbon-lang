"""
Test that dynamically discovered ivars of type IMP do not crash LLDB
"""




import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ObjCiVarIMPTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipIf(archs=['i386'])  # objc file does not build for i386
    @no_debug_info_test
    def test_imp_ivar_type(self):
        """Test that dynamically discovered ivars of type IMP do not crash LLDB"""
        self.build()
        exe = self.getBuildArtifact("a.out")

        # Create a target from the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Set up our breakpoint

        bkpt = lldbutil.run_break_set_by_source_regexp(self, "break here")

        # Now launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())

        self.assertEquals(process.GetState(), lldb.eStateStopped,
                        PROCESS_STOPPED)

        self.expect(
            'frame variable --ptr-depth=1 --show-types -d run -- object',
            substrs=[
                '(MyClass *) object = 0x',
                '(void *) myImp = 0x'])
        self.expect(
            'disassemble --start-address `((MyClass*)object)->myImp`',
            substrs=['-[MyClass init]'])
