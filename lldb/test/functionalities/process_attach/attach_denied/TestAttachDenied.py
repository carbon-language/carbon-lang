"""
Test denied process attach.
"""

import os
import shutil
import tempfile
import unittest2
import lldb
from lldbtest import *

exe_name = 'AttachDenied'  # Must match Makefile

class AttachDeniedTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipIfWindows
    def test_attach_to_process_by_id_denied(self):
        """Test attach by process id denied"""

        self.buildDefault()
        exe = os.path.join(os.getcwd(), exe_name)

        temp_dir = tempfile.mkdtemp()
        self.addTearDownHook(lambda: shutil.rmtree(temp_dir))

        # Use named pipe as a synchronization point between test and inferior.
        pid_pipe_path = os.path.join(temp_dir, "pid_pipe")
        os.mkfifo(pid_pipe_path)

        # Spawn a new process
        popen = self.spawnSubprocess(exe, [pid_pipe_path])
        self.addTearDownHook(self.cleanupSubprocesses)

        pid_pipe = open(pid_pipe_path, 'r')
        self.addTearDownHook(lambda: pid_pipe.close())
        pid = pid_pipe.read()

        self.expect('process attach -p ' + pid,
                    startstr = 'error: attach failed:',
                    error = True)


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
