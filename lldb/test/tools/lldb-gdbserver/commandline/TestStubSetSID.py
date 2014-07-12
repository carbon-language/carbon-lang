import unittest2

# Add the directory above ours to the python library path since we
# will import from there.
import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import gdbremote_testcase
import os
import select
import tempfile
import time
from lldbtest import *

class TestStubSetSIDTestCase(gdbremote_testcase.GdbRemoteTestCaseBase):
    def get_stub_sid(self, extra_stub_args=None):
        # Launch debugserver
        if extra_stub_args:
            self.debug_monitor_extra_args = extra_stub_args
        else:
            self.debug_monitor_extra_args = ""

        server = self.launch_debug_monitor()
        self.assertIsNotNone(server)
        self.assertTrue(server.isalive())
        server.expect("(debugserver|lldb-gdbserver)", timeout=10)

        # Get the process id for the stub.
        return os.getsid(server.pid)

    def sid_is_same_without_setsid(self):
        stub_sid = self.get_stub_sid()
        self.assertEquals(stub_sid, os.getsid(0))

    def sid_is_different_with_setsid(self):
        stub_sid = self.get_stub_sid(" --setsid")
        self.assertNotEquals(stub_sid, os.getsid(0))

    def sid_is_different_with_S(self):
        stub_sid = self.get_stub_sid(" -S")
        self.assertNotEquals(stub_sid, os.getsid(0))

    @debugserver_test
    @unittest2.expectedFailure() # This is the whole purpose of this feature, I would expect it to be the same without --setsid. Investigate.
    def test_sid_is_same_without_setsid_debugserver(self):
        self.init_debugserver_test()
        self.set_inferior_startup_launch()
        self.sid_is_same_without_setsid()

    @llgs_test
    @unittest2.expectedFailure() # This is the whole purpose of this feature, I would expect it to be the same without --setsid. Investigate.
    def test_sid_is_same_without_setsid_llgs(self):
        self.init_llgs_test()
        self.set_inferior_startup_launch()
        self.sid_is_same_without_setsid()

    @debugserver_test
    def test_sid_is_different_with_setsid_debugserver(self):
        self.init_debugserver_test()
        self.set_inferior_startup_launch()
        self.sid_is_different_with_setsid()

    @llgs_test
    def test_sid_is_different_with_setsid_llgs(self):
        self.init_llgs_test()
        self.set_inferior_startup_launch()
        self.sid_is_different_with_setsid()

    @debugserver_test
    def test_sid_is_different_with_S_debugserver(self):
        self.init_debugserver_test()
        self.set_inferior_startup_launch()
        self.sid_is_different_with_S()

    @llgs_test
    def test_sid_is_different_with_S_llgs(self):
        self.init_llgs_test()
        self.set_inferior_startup_launch()
        self.sid_is_different_with_S()
