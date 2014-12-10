"""Test Python APIs for process IO."""

import os, sys, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

class ProcessIOTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "dsym requires Darwin")
    @python_api_test
    @dsym_test
    def test_stdin_by_api_with_dsym(self):
        """Exercise SBProcess.PutSTDIN()."""
        self.buildDsym()
        self.do_stdin_by_api()

    @unittest2.skipIf(sys.platform.startswith("win32"), "stdio manipulation unsupported on Windows")
    @python_api_test
    @dwarf_test
    def test_stdin_by_api_with_dwarf(self):
        """Exercise SBProcess.PutSTDIN()."""
        self.buildDwarf()
        self.do_stdin_by_api()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "dsym requires Darwin")
    @python_api_test
    @dsym_test
    def test_stdin_redirection_with_dsym(self):
        """Exercise SBLaunchInfo::AddOpenFileAction() for STDIN without specifying STDOUT or STDERR."""
        self.buildDsym()
        self.do_stdin_redirection()

    @unittest2.skipIf(sys.platform.startswith("win32"), "stdio manipulation unsupported on Windows")
    @python_api_test
    @dwarf_test
    def test_stdin_redirection_with_dwarf(self):
        """Exercise SBLaunchInfo::AddOpenFileAction() for STDIN without specifying STDOUT or STDERR."""
        self.buildDwarf()
        self.do_stdin_redirection()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "dsym requires Darwin")
    @python_api_test
    @dsym_test
    def test_stdout_redirection_with_dsym(self):
        """Exercise SBLaunchInfo::AddOpenFileAction() for STDOUT without specifying STDIN or STDERR."""
        self.buildDsym()
        self.do_stdout_redirection()

    @unittest2.skipIf(sys.platform.startswith("win32"), "stdio manipulation unsupported on Windows")
    @python_api_test
    @dwarf_test
    def test_stdout_redirection_with_dwarf(self):
        """Exercise SBLaunchInfo::AddOpenFileAction() for STDOUT without specifying STDIN or STDERR."""
        self.buildDwarf()
        self.do_stdout_redirection()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "dsym requires Darwin")
    @python_api_test
    @dsym_test
    def test_stderr_redirection_with_dsym(self):
        """Exercise SBLaunchInfo::AddOpenFileAction() for STDERR without specifying STDIN or STDOUT."""
        self.buildDsym()
        self.do_stderr_redirection()

    @unittest2.skipIf(sys.platform.startswith("win32"), "stdio manipulation unsupported on Windows")
    @python_api_test
    @dwarf_test
    def test_stderr_redirection_with_dwarf(self):
        """Exercise SBLaunchInfo::AddOpenFileAction() for STDERR without specifying STDIN or STDOUT."""
        self.buildDwarf()
        self.do_stderr_redirection()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "dsym requires Darwin")
    @python_api_test
    @dsym_test
    def test_stdout_stderr_redirection_with_dsym(self):
        """Exercise SBLaunchInfo::AddOpenFileAction() for STDOUT and STDERR without redirecting STDIN."""
        self.buildDsym()
        self.do_stdout_stderr_redirection()

    @unittest2.skipIf(sys.platform.startswith("win32"), "stdio manipulation unsupported on Windows")
    @python_api_test
    @dwarf_test
    def test_stdout_stderr_redirection_with_dwarf(self):
        """Exercise SBLaunchInfo::AddOpenFileAction() for STDOUT and STDERR without redirecting STDIN."""
        self.buildDwarf()
        self.do_stdout_stderr_redirection()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Get the full path to our executable to be debugged.
        self.exe = os.path.join(os.getcwd(), "process_io")
        self.input_file  = os.path.join(os.getcwd(), "input.txt")
        self.output_file = os.path.join(os.getcwd(), "output.txt")
        self.error_file  = os.path.join(os.getcwd(), "error.txt")
        self.lines = ["Line 1", "Line 2", "Line 3"]
    
    def read_output_file_and_delete (self):
        self.assertTrue(os.path.exists(self.output_file), "Make sure output.txt file exists")
        f = open(self.output_file, 'r')
        contents = f.read()
        f.close()
        os.unlink(self.output_file)
        return contents

    def read_error_file_and_delete(self):
        self.assertTrue(os.path.exists(self.error_file), "Make sure error.txt file exists")
        f = open(self.error_file, 'r')
        contents = f.read()
        f.close()
        os.unlink(self.error_file)
        return contents

    def create_target(self):
        '''Create the target and launch info that will be used by all tests'''
        self.target = self.dbg.CreateTarget(self.exe)        
        self.launch_info = lldb.SBLaunchInfo([self.exe])
        self.launch_info.SetWorkingDirectory(self.get_process_working_directory())
    
    def redirect_stdin(self):
        '''Redirect STDIN (file descriptor 0) to use our input.txt file

        Make the input.txt file to use when redirecting STDIN, setup a cleanup action
        to delete the input.txt at the end of the test in case exceptions are thrown,
        and redirect STDIN in the launch info.'''
        f = open(self.input_file, 'w')
        for line in self.lines:
            f.write(line + "\n")
        f.close()
        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            os.unlink(self.input_file)
        
        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)
        self.launch_info.AddOpenFileAction(0, self.input_file, True, False);
        
    def redirect_stdout(self):
        '''Redirect STDOUT (file descriptor 1) to use our output.txt file'''
        self.launch_info.AddOpenFileAction(1, self.output_file, False, True);
    
    def redirect_stderr(self):
        '''Redirect STDERR (file descriptor 2) to use our error.txt file'''
        self.launch_info.AddOpenFileAction(2, self.error_file, False, True);
    
    def do_stdin_redirection(self):
        """Exercise SBLaunchInfo::AddOpenFileAction() for STDIN without specifying STDOUT or STDERR."""
        self.create_target()
        self.redirect_stdin()
        self.run_process(False)
        output = self.process.GetSTDOUT(1000)        
        self.check_process_output(output, output)

    def do_stdout_redirection(self):
        """Exercise SBLaunchInfo::AddOpenFileAction() for STDOUT without specifying STDIN or STDERR."""
        self.create_target()
        self.redirect_stdout()
        self.run_process(True)
        output = self.read_output_file_and_delete()
        error = self.process.GetSTDOUT(1000)
        self.check_process_output(output, error)

    def do_stderr_redirection(self):
        """Exercise SBLaunchInfo::AddOpenFileAction() for STDERR without specifying STDIN or STDOUT."""
        self.create_target()
        self.redirect_stderr()
        self.run_process(True)
        output = self.process.GetSTDOUT(1000)
        error = self.read_error_file_and_delete()
        self.check_process_output(output, error)

    def do_stdout_stderr_redirection(self):
        """Exercise SBLaunchInfo::AddOpenFileAction() for STDOUT and STDERR without redirecting STDIN."""
        self.create_target()
        self.redirect_stdout()
        self.redirect_stderr()
        self.run_process(True)
        output = self.read_output_file_and_delete()
        error = self.read_error_file_and_delete()
        self.check_process_output(output, error)

    def do_stdin_stdout_stderr_redirection(self):
        """Exercise SBLaunchInfo::AddOpenFileAction() for STDIN, STDOUT and STDERR."""
        # Make the input.txt file to use
        self.create_target()
        self.redirect_stdin()
        self.redirect_stdout()
        self.redirect_stderr()
        self.run_process(True)
        output = self.read_output_file_and_delete()
        error = self.read_error_file_and_delete()
        self.check_process_output(output, error)
        
    def do_stdin_by_api(self):
        """Launch a process and use SBProcess.PutSTDIN() to write data to it."""
        self.create_target()
        self.run_process(True)
        output = self.process.GetSTDOUT(1000)
        self.check_process_output(output, output)
        
    def run_process(self, put_stdin):
        '''Run the process to completion and optionally put lines to STDIN via the API if "put_stdin" is True'''
        # Set the breakpoints
        self.breakpoint = self.target.BreakpointCreateBySourceRegex('Set breakpoint here', lldb.SBFileSpec("main.c"))
        self.assertTrue(self.breakpoint.GetNumLocations() > 0, VALID_BREAKPOINT)

        # Launch the process, and do not stop at the entry point.
        error = lldb.SBError()
        # This should launch the process and it should exit by the time we get back
        # because we have synchronous mode enabled
        self.process = self.target.Launch (self.launch_info, error)

        self.assertTrue(error.Success(), "Make sure process launched successfully")
        self.assertTrue(self.process, PROCESS_IS_VALID)

        if self.TraceOn():
            print "process launched."

        # Frame #0 should be at our breakpoint.
        threads = lldbutil.get_threads_stopped_at_breakpoint (self.process, self.breakpoint)
        
        self.assertTrue(len(threads) == 1)
        self.thread = threads[0]
        self.frame = self.thread.frames[0]
        self.assertTrue(self.frame, "Frame 0 is valid.")

        if self.TraceOn():
            print "process stopped at breakpoint, sending STDIN via LLDB API."

        # Write data to stdin via the public API if we were asked to
        if put_stdin:
            for line in self.lines:
                self.process.PutSTDIN(line + "\n")
        
        # Let process continue so it will exit
        self.process.Continue()
        state = self.process.GetState()
        self.assertTrue(state == lldb.eStateExited, PROCESS_IS_VALID)
        
    def check_process_output (self, output, error):
            # Since we launched the process without specifying stdin/out/err,
            # a pseudo terminal is used for stdout/err, and we are satisfied
            # once "input line=>1" appears in stdout.
            # See also main.c.
        if self.TraceOn():
            print "output = '%s'" % output
            print "error = '%s'" % error
        
        for line in self.lines:
            check_line = 'input line to stdout: %s' % (line)
            self.assertTrue(check_line in output, "verify stdout line shows up in STDOUT")
        for line in self.lines:
            check_line = 'input line to stderr: %s' % (line)
            self.assertTrue(check_line in error, "verify stderr line shows up in STDERR")

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
