"""
Test lldb-gdbserver operation
"""

import unittest2
import pexpect
import sys
from lldbtest import *
from lldbgdbserverutils import *

class LldbGdbServerTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    port = 12345

    def setUp(self):
        TestBase.setUp(self)
        self.lldb_gdbserver_exe = get_lldb_gdbserver_exe()
        if not self.lldb_gdbserver_exe:
            self.skipTest("lldb_gdbserver exe not specified")

    def test_exe_starts(self):
        # start the server
        server = pexpect.spawn("{} localhost:{}".format(self.lldb_gdbserver_exe, self.port))

        # Turn on logging for what the child sends back.
        if self.TraceOn():
            server.logfile_read = sys.stdout

        # Schedule lldb-gdbserver to be shutting down during teardown.
        def shutdown_lldb_gdbserver():
            server.close()
        self.addTearDownHook(shutdown_lldb_gdbserver)

        # Wait until we receive the server ready message before continuing.
        server.expect_exact('Listening for a connection on localhost:{}'.format(self.port))

if __name__ == '__main__':
    unittest2.main()
