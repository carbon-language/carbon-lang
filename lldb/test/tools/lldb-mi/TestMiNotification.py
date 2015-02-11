"""
Test that the lldb-mi driver nofities user properly.
"""

import lldbmi_testcase
from lldbtest import *
import unittest2

class MiNotificationTestCase(lldbmi_testcase.MiTestCaseBase):

    @lldbmi_test
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    @unittest2.skip("lldb-mi doesn't print prompt in all cases")
    def test_lldbmi_prompt(self):
        """Test that 'lldb-mi --interpreter' echos '(gdb)' after commands and events."""

        self.spawnLldbMi(args = None)

        # Test that lldb-mi is ready after startup
        self.expect(self.child_prompt, exactly = True)

        # Test that lldb-mi is ready after -file-exec-and-symbols
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")
        self.expect(self.child_prompt, exactly = True)

        # Test that lldb-mi is ready after -break-insert
        self.runCmd("-break-insert -f b_MyFunction")
        self.expect("\^done,bkpt={number=\"1\"")
        self.expect(self.child_prompt, exactly = True)

        # Test that lldb-mi is ready after -exec-run
        self.runCmd("-exec-run")
        self.expect("\*running")
        self.expect(self.child_prompt, exactly = True)

        # Test that lldb-mi is ready after BP hit
        self.expect("\*stopped,reason=\"breakpoint-hit\"")
        self.expect(self.child_prompt, exactly = True)

        # Test that lldb-mi is ready after -exec-continue
        self.runCmd("-exec-continue")
        self.expect("\^running")
        self.expect(self.child_prompt, exactly = True)

        # Test that lldb-mi is ready after program exited
        self.expect("\*stopped,reason=\"exited-normally\"")
        self.expect(self.child_prompt, exactly = True)

    @lldbmi_test
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    def test_lldbmi_stopped_when_stopatentry_local(self):
        """Test that 'lldb-mi --interpreter' notifies after it was stopped on entry (local)."""

        self.spawnLldbMi(args = None)

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # Run with stop-at-entry flag
        self.runCmd("-interpreter-exec command \"process launch -s\"")
        self.expect("\^done")

        # Test that *stopped is printed
        # Note that message is different in Darwin and Linux:
        # Darwin: "*stopped,reason=\"signal-received\",signal=\"17\",thread-id=\"1\",stopped-threads=\"all\"" 
        # Linux:  "*stopped,reason=\"end-stepping-range\",frame={addr=\"0x[0-9a-f]+\",func=\"??\",args=\[\],file=\"??\",fullname=\"??\",line=\"-1\"},thread-id=\"1\",stopped-threads=\"all\"
        self.expect("\*stopped,reason=\"(signal-received|end-stepping-range)\",.+,thread-id=\"1\",stopped-threads=\"all\"")
        
        # Run to main to make sure we have not exited the application
        self.runCmd("-break-insert -f main")
        self.expect("\^done,bkpt={number=\"1\"")
        self.runCmd("-exec-continue")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

    @lldbmi_test
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_lldbmi_stopped_when_stopatentry_remote(self):
        """Test that 'lldb-mi --interpreter' notifies after it was stopped on entry (remote)."""

        # Prepare debugserver
        import os, sys
        lldb_gdbserver_folder = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), "lldb-gdbserver"))
        sys.path.append(lldb_gdbserver_folder)
        import lldbgdbserverutils
        debugserver_exe = lldbgdbserverutils.get_debugserver_exe()
        if not debugserver_exe:
            raise Exception("debugserver not found")
        hostname = "localhost"
        import random
        port = 12000 + random.randint(0,3999) # the same as GdbRemoteTestCaseBase.get_next_port
        import pexpect
        debugserver_child = pexpect.spawn("%s %s:%d" % (debugserver_exe, hostname, port))

        self.spawnLldbMi(args = None)

        # Connect to debugserver
        self.runCmd("-interpreter-exec command \"platform select remote-macosx --sysroot /\"")
        self.expect("\^done")
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")
        self.runCmd("-interpreter-exec command \"process connect connect://%s:%d\"" % (hostname, port))
        self.expect("\^done")

        try:
            # Run with stop-at-entry flag
            self.runCmd("-interpreter-exec command \"process launch -s\"")
            self.expect("\^done")

            # Test that *stopped is printed
            self.expect("\*stopped,reason=\"signal-received\",signal=\"17\",thread-id=\"1\",stopped-threads=\"all\"")

            # Exit
            self.runCmd("-gdb-exit")
            self.expect("\^exit")

        finally:
            # Clean up
            debugserver_child.terminate(force = True)

    @lldbmi_test
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    def test_lldbmi_stopped_when_segfault_local(self):
        """Test that 'lldb-mi --interpreter' notifies after it was stopped when segfault occurred (local)."""

        self.spawnLldbMi(args = None)

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # Run to main
        self.runCmd("-break-insert -f main")
        self.expect("\^done,bkpt={number=\"1\"")
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Set dosegfault=1 and run (to cause a segfault error)
        self.runCmd("-data-evaluate-expression \"dosegfault=1\"")
        self.expect("\^done,value=\"1\"")
        self.runCmd("-exec-continue")
        self.expect("\^running")

        # Test that *stopped is printed
        self.expect("\*stopped,reason=\"exception-received\",exception=\"EXC_BAD_ACCESS \(code=1, address=0x0\)\",thread-id=\"1\",stopped-threads=\"all\"")

    @lldbmi_test
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_lldbmi_stopped_when_segfault_remote(self):
        """Test that 'lldb-mi --interpreter' notifies after it was stopped when segfault occurred (remote)."""

        # Prepare debugserver
        import os, sys
        lldb_gdbserver_folder = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), "lldb-gdbserver"))
        sys.path.append(lldb_gdbserver_folder)
        import lldbgdbserverutils
        debugserver_exe = lldbgdbserverutils.get_debugserver_exe()
        if not debugserver_exe:
            raise Exception("debugserver not found")
        hostname = "localhost"
        import random
        port = 12000 + random.randint(0,3999) # the same as GdbRemoteTestCaseBase.get_next_port
        import pexpect
        debugserver_child = pexpect.spawn("%s %s:%d" % (debugserver_exe, hostname, port))

        self.spawnLldbMi(args = None)

        # Connect to debugserver
        self.runCmd("-interpreter-exec command \"platform select remote-macosx --sysroot /\"")
        self.expect("\^done")
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")
        self.runCmd("-interpreter-exec command \"process connect connect://%s:%d\"" % (hostname, port))
        self.expect("\^done")

        try:
            # Run to main
            self.runCmd("-break-insert -f main")
            self.expect("\^done,bkpt={number=\"1\"")
            #FIXME -exec-run doesn't work
            self.runCmd("-interpreter-exec command \"process launch\"") #FIXME: self.runCmd("-exec-run")
            self.expect("\^done")                                       #FIXME: self.expect("\^running")
            self.expect("\*stopped,reason=\"breakpoint-hit\"")

            # Set dosegfault=1 and run (to cause a segfault error)
            self.runCmd("-data-evaluate-expression \"dosegfault=1\"")
            self.expect("\^done,value=\"1\"")
            self.runCmd("-exec-continue")
            self.expect("\^running")

            # Test that *stopped is printed
            self.expect("\*stopped,reason=\"exception-received\",exception=\"EXC_BAD_ACCESS \(code=1, address=0x0\)\",thread-id=\"1\",stopped-threads=\"all\"")

            # Exit
            self.runCmd("-gdb-exit")
            self.expect("\^exit")

        finally:
            # Clean up
            debugserver_child.terminate(force = True)

if __name__ == '__main__':
    unittest2.main()
