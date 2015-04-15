import unittest2

import gdbremote_testcase
from lldbtest import *

class TestGdbRemoteSingleStep(gdbremote_testcase.GdbRemoteTestCaseBase):

    @debugserver_test
    @dsym_test
    def test_single_step_only_steps_one_instruction_with_s_debugserver_dsym(self):
        self.init_debugserver_test()
        self.buildDsym()
        self.set_inferior_startup_launch()
        self.single_step_only_steps_one_instruction(use_Hc_packet=True, step_instruction="s")

    @llgs_test
    @dwarf_test
    def test_single_step_only_steps_one_instruction_with_s_llgs_dwarf(self):
        self.init_llgs_test()
        self.buildDwarf()
        self.set_inferior_startup_launch()
        self.single_step_only_steps_one_instruction(use_Hc_packet=True, step_instruction="s")

if __name__ == '__main__':
    unittest2.main()
