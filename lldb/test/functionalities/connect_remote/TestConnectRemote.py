"""
Test lldb 'process connect' command.
"""

import os
import unittest2
import lldb
import re
from lldbtest import *

class ConnectRemoteTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def test_connect_remote(self):
        """Test "process connect connect:://localhost:[port]"."""

        import pexpect
        # First, we'll start a fake debugserver (a simple echo server).
        fakeserver = pexpect.spawn('./EchoServer.py')

        # Turn on logging for what the child sends back.
        if self.TraceOn():
            fakeserver.logfile_read = sys.stdout

        # Schedule the fake debugserver to be shutting down during teardown.
        def shutdown_fakeserver():
            fakeserver.close()
        self.addTearDownHook(shutdown_fakeserver)

        # Wait until we receive the server ready message before continuing.
        line = fakeserver.readline()
        self.assertTrue(line.startswith("Listening on localhost:"))
        port = int(re.match('Listening on localhost:([0-9]+)', line).group(1))
        self.assertTrue(port > 0)

        # Connect to the fake server....
        self.runCmd("process connect -p gdb-remote connect://localhost:" + str(port))


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
