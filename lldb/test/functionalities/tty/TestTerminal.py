"""
Test lldb command aliases.
"""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

class LaunchInTerminalTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    # Darwin is the only platform that I know of that supports optionally launching
    # a program in a separate terminal window. It would be great if other platforms
    # added support for this.
    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_launch_in_terminal (self):
        exe = "/bin/ls"
        target = self.dbg.CreateTarget(exe)
        launch_info = lldb.SBLaunchInfo(["-lAF", "/tmp/"])
        launch_info.SetLaunchFlags(lldb.eLaunchFlagLaunchInTTY)
        error = lldb.SBError()
        process = target.Launch (launch_info, error)
        self.assertTrue(error.Success(), "Make sure launch happened successfully in a terminal window")
        # Running in synchronous mode our process should have run and already exited by the time target.Launch() returns
        self.assertTrue(process.GetState() == lldb.eStateExited)

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()

