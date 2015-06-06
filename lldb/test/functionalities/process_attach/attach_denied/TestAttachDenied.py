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
        return (err, shell_command.GetStatus(), shell_command.GetOutput())

    @skipIfWindows
    def test_attach_to_process_by_id_denied(self):
        """Test attach by process id denied"""

        self.buildDefault()
        exe = os.path.join(os.getcwd(), exe_name)

        # Use a file as a synchronization point between test and inferior.
        pid_file_path = lldbutil.append_to_process_working_directory(
                "pid_file_%d" % (int(time.time())))
        self.addTearDownHook(lambda: self.run_platform_command("rm %s" % (pid_file_path)))

        # Spawn a new process
        popen = self.spawnSubprocess(exe, [pid_file_path])
        self.addTearDownHook(self.cleanupSubprocesses)

        max_attempts = 5
        for i in range(max_attempts):
            err, retcode, msg = self.run_platform_command("ls %s" % pid_file_path)
            if err.Success() and retcode == 0:
                break
            else:
                print msg
            if i < max_attempts:
                # Exponential backoff!
                time.sleep(pow(2, i) * 0.25)
        else:
            self.fail("Child PID file %s not found even after %d attempts." % (pid_file_path, max_attempts))
        err, retcode, pid = self.run_platform_command("cat %s" % (pid_file_path))
        self.assertTrue(err.Success() and retcode == 0,
                        "Failed to read file %s: %s, retcode: %d" % (pid_file_path, err.GetCString(), retcode))

        self.expect('process attach -p ' + pid,
                    startstr = 'error: attach failed:',
                    error = True)


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
