"""
Test denied process attach.
"""

import os
import time
import unittest2
import lldb
from lldbtest import *

exe_name = 'AttachDenied'  # Must match Makefile

class AttachDeniedTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def run_platform_command(self, cmd):
        platform = self.dbg.GetSelectedPlatform()
        shell_command = lldb.SBPlatformShellCommand(cmd)
        err = platform.Run(shell_command)
        return (err, shell_command.GetOutput())

    @skipIfWindows
    def test_attach_to_process_by_id_denied(self):
        """Test attach by process id denied"""

        self.buildDefault()
        exe = os.path.join(os.getcwd(), exe_name)

        # Use named pipe as a synchronization point between test and inferior.
        pid_pipe_path = os.path.join(self.get_process_working_directory(),
                                     "pid_pipe_%d" % (int(time.time())))

        err, _ = self.run_platform_command("mkfifo %s" % (pid_pipe_path))
        self.assertTrue(err.Success(), "Failed to create FIFO %s: %s" % (pid_pipe_path, err.GetCString()))

        self.addTearDownHook(lambda: self.run_platform_command("rm %s" % (pid_pipe_path)))

        # Spawn a new process
        popen = self.spawnSubprocess(exe, [pid_pipe_path])
        self.addTearDownHook(self.cleanupSubprocesses)

        err, pid = self.run_platform_command("cat %s" % (pid_pipe_path))
        self.assertTrue(err.Success(), "Failed to read FIFO %s: %s" % (pid_pipe_path, err.GetCString()))

        self.expect('process attach -p ' + pid,
                    startstr = 'error: attach failed:',
                    error = True)


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
