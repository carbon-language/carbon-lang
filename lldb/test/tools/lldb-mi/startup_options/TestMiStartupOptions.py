"""
Test lldb-mi startup options.
"""

import lldbmi_testcase
from lldbtest import *
import unittest2

class MiStartupOptionsTestCase(lldbmi_testcase.MiTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)

    @lldbmi_test
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    @skipIfFreeBSD # llvm.org/pr22411: Failure presumably due to known thread races
    def test_lldbmi_executable_option_file(self):
        """Test that 'lldb-mi --interpreter %s' loads executable file."""

        self.spawnLldbMi(args = "%s" % self.myexe)

        # Test that lldb-mi is ready after startup
        self.expect(self.child_prompt, exactly = True)

        # Test that the executable is loaded when file was specified
        self.expect("-file-exec-and-symbols \"%s\"" % self.myexe)
        self.expect("\^done")

        # Test that lldb-mi is ready when executable was loaded
        self.expect(self.child_prompt, exactly = True)

        # Run
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"exited-normally\"")

    @lldbmi_test
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    @skipIfFreeBSD # llvm.org/pr22411: Failure presumably due to known thread races
    def test_lldbmi_executable_option_unknown_file(self):
        """Test that 'lldb-mi --interpreter %s' fails on unknown executable file."""

        # Prepare path to executable
        path = "unknown_file"

        self.spawnLldbMi(args = "%s" % path)

        # Test that lldb-mi is ready after startup
        self.expect(self.child_prompt, exactly = True)

        # Test that the executable isn't loaded when unknown file was specified
        self.expect("-file-exec-and-symbols \"%s\"" % path)
        self.expect("\^error,msg=\"Command 'file-exec-and-symbols'. Target binary '%s' is invalid. error: unable to find executable for '%s'\"" % (path, path))

        # Test that lldb-mi is ready when executable was loaded
        self.expect(self.child_prompt, exactly = True)

    @lldbmi_test
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    @skipIfFreeBSD # llvm.org/pr22411: Failure presumably due to known thread races
    def test_lldbmi_executable_option_absolute_path(self):
        """Test that 'lldb-mi --interpreter %s' loads executable which is specified via absolute path."""

        # Prepare path to executable
        import os
        path = os.path.join(os.getcwd(), self.myexe)

        self.spawnLldbMi(args = "%s" % path)

        # Test that lldb-mi is ready after startup
        self.expect(self.child_prompt, exactly = True)

        # Test that the executable is loaded when file was specified using absolute path
        self.expect("-file-exec-and-symbols \"%s\"" % path)
        self.expect("\^done")

        # Test that lldb-mi is ready when executable was loaded
        self.expect(self.child_prompt, exactly = True)

        # Run
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"exited-normally\"")

    @lldbmi_test
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    @skipIfFreeBSD # llvm.org/pr22411: Failure presumably due to known thread races
    def test_lldbmi_executable_option_relative_path(self):
        """Test that 'lldb-mi --interpreter %s' loads executable which is specified via relative path."""

        # Prepare path to executable
        path = "./%s" % self.myexe

        self.spawnLldbMi(args = "%s" % path)

        # Test that lldb-mi is ready after startup
        self.expect(self.child_prompt, exactly = True)

        # Test that the executable is loaded when file was specified using relative path
        self.expect("-file-exec-and-symbols \"%s\"" % path)
        self.expect("\^done")

        # Test that lldb-mi is ready when executable was loaded
        self.expect(self.child_prompt, exactly = True)

        # Run
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"exited-normally\"")

    @lldbmi_test
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    @skipIfFreeBSD # llvm.org/pr22411: Failure presumably due to known thread races
    def test_lldbmi_executable_option_unknown_path(self):
        """Test that 'lldb-mi --interpreter %s' fails on executable file which is specified via unknown path."""

        # Prepare path to executable
        path = "unknown_dir/%s" % self.myexe

        self.spawnLldbMi(args = "%s" % path)

        # Test that lldb-mi is ready after startup
        self.expect(self.child_prompt, exactly = True)

        # Test that the executable isn't loaded when file was specified using unknown path
        self.expect("-file-exec-and-symbols \"%s\"" % path)
        self.expect("\^error,msg=\"Command 'file-exec-and-symbols'. Target binary '%s' is invalid. error: unable to find executable for '%s'\"" % (path, path))

        # Test that lldb-mi is ready when executable was loaded
        self.expect(self.child_prompt, exactly = True)

if __name__ == '__main__':
    unittest2.main()
