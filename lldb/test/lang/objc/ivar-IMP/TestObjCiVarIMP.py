"""
Test that dynamically discovered ivars of type IMP do not crash LLDB
"""

import os, time
import re
import unittest2
import lldb, lldbutil
from lldbtest import *
import commands

def execute_command (command):
    # print '%% %s' % (command)
    (exit_status, output) = commands.getstatusoutput (command)
    # if output:
    #     print output
    # print 'status = %u' % (exit_status)
    return exit_status

class ObjCiVarIMPTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipUnlessDarwin
    def test_imp_ivar_type(self):
        """Test that dynamically discovered ivars of type IMP do not crash LLDB"""
        if self.getArchitecture() == 'i386':
            # rdar://problem/9946499
            self.skipTest("Dynamic types for ObjC V1 runtime not implemented")
        self.buildReproCase()
        self.runTheTest()

    def setUp(self):
        # Call super's setUp().                                                                                                           
        TestBase.setUp(self)

    def buildReproCase (self):
        execute_command("make repro")

    def runTheTest(self):
        """MakeTest that dynamically discovered ivars of type IMP do not crash LLDB"""
        def cleanup():
            execute_command("make cleanup")
        self.addTearDownHook(cleanup)

        exe = os.path.join(os.getcwd(), "a.out")

        # Create a target from the debugger.

        target = self.dbg.CreateTarget (exe)
        self.assertTrue(target, VALID_TARGET)

        # Set up our breakpoint

        bkpt = lldbutil.run_break_set_by_source_regexp (self, "break here")

        # Now launch the process, and do not stop at the entry point.
        process = target.LaunchSimple (None, None, self.get_process_working_directory())

        self.assertTrue(process.GetState() == lldb.eStateStopped,
                        PROCESS_STOPPED)

        self.expect('frame variable --ptr-depth=1 --show-types -d run -- object', substrs=[
            '(MyClass *) object = 0x',
            '(void *) myImp = 0x'
        ])
        self.expect('disassemble --start-address `((MyClass*)object)->myImp`', substrs=[
            '-[MyClass init]'
        ])

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
