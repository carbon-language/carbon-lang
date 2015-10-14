"""
Tests the binary ($x) and hex ($m) memory read packets of the remote stub
"""

import os
import unittest2
import lldb
from lldbtest import *
import lldbutil
import binascii


class MemoryReadTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipUnlessPlatform(getDarwinOSTriples()+["linux"])
    def test_memory_read(self):
        self.build()
        exe = os.path.join (os.getcwd(), "a.out")

        target = self.dbg.CreateTarget(exe)
        lldbutil.run_break_set_by_symbol(self, "main")

        process = target.LaunchSimple (None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)
        self.assertEqual(process.GetState(), lldb.eStateStopped, "Process is stopped")

        pc = process.GetSelectedThread().GetSelectedFrame().GetPC()
        for size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
            error = lldb.SBError()
            memory = process.ReadMemory(pc, size, error)
            self.assertTrue(error.Success())
            self.match("process plugin packet send x%x,%x" % (pc, size), ["response:", memory])
            self.match("process plugin packet send m%x,%x" % (pc, size), ["response:", binascii.hexlify(memory)])

        process.Continue()
        self.assertEqual(process.GetState(), lldb.eStateExited, "Process exited")


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
