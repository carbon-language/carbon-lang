"""Test that we handle inferiors which change their process group"""

import os
import unittest2
import lldb
from lldbtest import *
import lldbutil


class ChangeProcessGroupTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break for main.c.
        self.line = line_number('main.c', '// Set breakpoint here')

    @skipIfWindows # setpgid call does not exist on Windows
    @skipUnlessDarwin
    @dsym_test
    def test_setpgid_with_dsym(self):
        self.buildDsym()
        self.setpgid()

    @skipIfWindows # setpgid call does not exist on Windows
    @dwarf_test
    def test_setpgid_with_dwarf(self):
        self.buildDwarf()
        self.setpgid()

    def run_platform_command(self, cmd):
        platform = self.dbg.GetSelectedPlatform()
        shell_command = lldb.SBPlatformShellCommand(cmd)
        err = platform.Run(shell_command)
        return (err, shell_command.GetStatus(), shell_command.GetOutput())

    def setpgid(self):
        exe = os.path.join(os.getcwd(), 'a.out')

        # Use a file as a synchronization point between test and inferior.
        pid_file_path = os.path.join(self.get_process_working_directory(),
                                     "pid_file_%d" % (int(time.time())))
        self.addTearDownHook(lambda: self.run_platform_command("rm %s" % (pid_file_path)))

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


        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        listener = lldb.SBListener("my.attach.listener")
        error = lldb.SBError()
        process = target.AttachToProcessWithID(listener, int(pid), error)
        self.assertTrue(error.Success() and process, PROCESS_IS_VALID)

        # set a breakpoint just before the setpgid() call
        lldbutil.run_break_set_by_file_and_line(self, 'main.c', self.line, num_expected_locations=-1)

        thread = process.GetSelectedThread()
        # this gives a chance for the thread to exit the sleep syscall and sidesteps
        # <https://llvm.org/bugs/show_bug.cgi?id=23659> on linux
        thread.StepInstruction(False)

        # release the child from its loop
        self.expect("expr release_child_flag = 1", substrs = ["= 1"])
        process.Continue()

        # make sure the child's process group id is different from its pid
        self.expect("print (int)getpgid(0)", substrs = [pid], matching=False)

        # step over the setpgid() call
        thread.StepOver()
        self.assertEqual(thread.GetStopReason(), lldb.eStopReasonPlanComplete)

        # verify that the process group has been set correctly
        # this also checks that we are still in full control of the child
        self.expect("print (pid_t)getpgid(0)", substrs = [pid])

        # run to completion
        process.Continue()
        self.assertEqual(process.GetState(), lldb.eStateExited)

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
