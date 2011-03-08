"""
Test lldb 'process connect' command.
"""

import os, time
import unittest2
import lldb
from lldbtest import *

class ConnectRemoteTestCase(TestBase):

    mydir = "connect_remote"

    @unittest2.expectedFailure
    def test_connect_remote(self):
        """Test "process connect connect:://localhost:12345"."""

        # First, we'll start a fake debugserver (a simple echo server).
        import subprocess
        fakeserver = subprocess.Popen('./EchoServer.py')
        # This does the cleanup afterwards.
        def cleanup_fakeserver():
            fakeserver.kill()
            fakeserver.wait()
        self.addTearDownHook(cleanup_fakeserver)

        self.runCmd("process connect connect://localhost:12345")


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
