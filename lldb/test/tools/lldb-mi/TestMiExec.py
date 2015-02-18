"""
Test that the lldb-mi driver works with -exec-xxx commands
"""

import lldbmi_testcase
from lldbtest import *
import unittest2

class MiExecTestCase(lldbmi_testcase.MiTestCaseBase):

    @lldbmi_test
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    @unittest2.skip("-exec-abort isn't implemented")
    def test_lldbmi_exec_abort(self):
        """Test that 'lldb-mi --interpreter' works for -exec-abort."""

        self.spawnLldbMi(args = None)

        # Test that -exec-abort fails on invalid process
        self.runCmd("-exec-abort")
        self.expect("\^error,msg=\"Command 'exec-abort'. Invalid process during debug session\"")

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # Set arguments
        self.runCmd("-exec-arguments arg1")
        self.expect("\^done")

        # Run to main
        self.runCmd("-break-insert -f main")
        self.expect("\^done,bkpt={number=\"1\"")
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Test that arguments were passed
        self.runCmd("-data-evaluate-expression argc")
        self.expect("\^done,value=\"2\"")

        # Test that program may be aborted
        self.runCmd("-exec-abort")
        self.expect("\^done")
        self.expect("\*stopped,reason=\"exited-normally\"")

        # Test that program can be run again
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Test that arguments were passed again
        self.runCmd("-data-evaluate-expression argc")
        self.expect("\^done,value=\"2\"")

        # Test that program may be aborted again
        self.runCmd("-exec-abort")
        self.expect("\^done")
        self.expect("\*stopped,reason=\"exited-normally\"")

    @lldbmi_test
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    @skipIfFreeBSD # llvm.org/pr22411: Failure presumably due to known thread races
    @skipIfLinux # llvm.org/pr22411: Failure presumably due to known thread races
    def test_lldbmi_exec_arguments_set(self):
        """Test that 'lldb-mi --interpreter' can pass args using -exec-arguments."""

        self.spawnLldbMi(args = None)

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # Set arguments
        self.runCmd("-exec-arguments --arg1 \"2nd arg\" third_arg fourth=\"4th arg\"")
        self.expect("\^done")

        # Run to main
        self.runCmd("-break-insert -f main")
        self.expect("\^done,bkpt={number=\"1\"")
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Check argc and argv to see if arg passed
        self.runCmd("-data-evaluate-expression argc")
        self.expect("\^done,value=\"5\"")
        #self.runCmd("-data-evaluate-expression argv[1]")
        #self.expect("\^done,value=\"--arg1\"")
        self.runCmd("-interpreter-exec command \"print argv[1]\"")
        self.expect("\"--arg1\"")
        #self.runCmd("-data-evaluate-expression argv[2]")
        #self.expect("\^done,value=\"2nd arg\"")
        self.runCmd("-interpreter-exec command \"print argv[2]\"")
        #FIXME: lldb-mi doesn't handle inner quotes
        self.expect("\"\\\\\\\"2nd arg\\\\\\\"\"") #FIXME: self.expect("\"2nd arg\"")
        #self.runCmd("-data-evaluate-expression argv[3]")
        #self.expect("\^done,value=\"third_arg\"")
        self.runCmd("-interpreter-exec command \"print argv[3]\"")
        self.expect("\"third_arg\"")
        #self.runCmd("-data-evaluate-expression argv[4]")
        #self.expect("\^done,value=\"fourth=\\\\\\\"4th arg\\\\\\\"\"")
        self.runCmd("-interpreter-exec command \"print argv[4]\"")
        self.expect("\"fourth=\\\\\\\"4th arg\\\\\\\"\"")

    @lldbmi_test
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    def test_lldbmi_exec_arguments_reset(self):
        """Test that 'lldb-mi --interpreter' can reset previously set args using -exec-arguments."""

        self.spawnLldbMi(args = None)

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # Set arguments
        self.runCmd("-exec-arguments foo bar baz")
        self.expect("\^done")
        self.runCmd("-exec-arguments")
        self.expect("\^done")

        # Run to main
        self.runCmd("-break-insert -f main")
        self.expect("\^done,bkpt={number=\"1\"")
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Check argc to see if arg passed
        self.runCmd("-data-evaluate-expression argc")
        self.expect("\^done,value=\"1\"")

    @lldbmi_test
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    def test_lldbmi_exec_next(self):
        """Test that 'lldb-mi --interpreter' works for stepping."""

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

        # Warning: the following is sensative to the lines in the source.

        # Test -exec-next
        self.runCmd("-exec-next --thread 1 --frame 0")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"end-stepping-range\".*main.c\",line=\"26\"")

        # Test that --thread is optional
        self.runCmd("-exec-next --frame 0")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"end-stepping-range\".*main.c\",line=\"27\"")

        # Test that --frame is optional
        self.runCmd("-exec-next --thread 1")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"end-stepping-range\".*main.c\",line=\"29\"")

        # Test that both --thread and --frame are optional
        self.runCmd("-exec-next --thread 1")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"end-stepping-range\".*main.c\",line=\"31\"")

        # Test that an invalid --thread is handled
        self.runCmd("-exec-next --thread 0")
        self.expect("\^error,message=\"error: Thread index 0 is out of range")
        self.runCmd("-exec-next --thread 10")
        self.expect("\^error,message=\"error: Thread index 10 is out of range")

        # Test that an invalid --frame is handled
        # FIXME: no error is returned
        self.runCmd("-exec-next --frame 10")
        #self.expect("\^error: Frame index 10 is out of range")

    @lldbmi_test
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    def test_lldbmi_exec_next_instruction(self):
        """Test that 'lldb-mi --interpreter' works for instruction stepping."""

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

        # Warning: the following is sensative to the lines in the
        # source and optimizations

        # Test -exec-next-instruction
        self.runCmd("-exec-next-instruction --thread 1 --frame 0")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"end-stepping-range\".*main.c\",line=\"24\"")

        # Test that --thread is optional
        self.runCmd("-exec-next-instruction --frame 0")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"end-stepping-range\".*main.c\",line=\"24\"")

        # Test that --frame is optional
        self.runCmd("-exec-next-instruction --thread 1")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"end-stepping-range\".*main.c\",line=\"24\"")

        # Test that both --thread and --frame are optional
        self.runCmd("-exec-next-instruction --thread 1")
        self.expect("\^running")
        # Depending on compiler, it can stop at different line.
        self.expect("\*stopped,reason=\"end-stepping-range\".*main.c\",line=\"2[4-6]\"")

        # Test that an invalid --thread is handled
        self.runCmd("-exec-next-instruction --thread 0")
        self.expect("\^error,message=\"error: Thread index 0 is out of range")
        self.runCmd("-exec-next-instruction --thread 10")
        self.expect("\^error,message=\"error: Thread index 10 is out of range")

        # Test that an invalid --frame is handled
        # FIXME: no error is returned
        self.runCmd("-exec-next-instruction --frame 10")
        #self.expect("\^error: Frame index 10 is out of range")

    @lldbmi_test
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    @skipIfFreeBSD # llvm.org/pr22411: Failure presumably due to known thread races
    @skipIfLinux # llvm.org/pr22411: Failure presumably due to known thread races
    def test_lldbmi_exec_step(self):
        """Test that 'lldb-mi --interpreter' works for stepping into."""

        self.spawnLldbMi(args = None)

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # Run to printf call
        line = line_number('main.c', '//BP_printf_call')
        self.runCmd("-break-insert -f main.c:%d" % line)
        self.expect("\^done,bkpt={number=\"1\"")
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Warning: the following is sensative to the lines in the source

        # Test that -exec-step does not step into printf (which
        # has no debug info)
        #FIXME: is this supposed to step into printf?
        self.runCmd("-exec-step --thread 1 --frame 0")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"end-stepping-range\".*main.c\",line=\"26\"")

        # Test that -exec-step steps into a_MyFunction and back out
        # (and that --thread is optional)
        self.runCmd("-exec-step --frame 0")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"end-stepping-range\".*func=\"a_MyFunction\"")
        #FIXME: is this supposed to step into printf?
        self.runCmd("-exec-step --frame 0")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"end-stepping-range\".*func=\"a_MyFunction\"")
        # Use -exec-finish here to make sure that control reaches the caller.
        # -exec-step can keep us in the a_MyFunction for gcc
        self.runCmd("-exec-finish --frame 0")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"end-stepping-range\".*main.c\",line=\"26\"")
        self.runCmd("-exec-step --frame 0")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"end-stepping-range\".*main.c\",line=\"27\"")

        # Test that -exec-step steps into b_MyFunction
        # (and that --frame is optional)
        self.runCmd("-exec-step --thread 1")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"end-stepping-range\".*func=\"b_MyFunction\"")

        # Test that -exec-step steps into a_MyFunction from inside
        # b_MyFunction (and that both --thread and --frame are optional)
        self.runCmd("-exec-step")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"end-stepping-range\".*func=\"a_MyFunction\"")

        # Test that an invalid --thread is handled
        self.runCmd("-exec-step --thread 0")
        self.expect("\^error,message=\"error: Thread index 0 is out of range")
        self.runCmd("-exec-step --thread 10")
        self.expect("\^error,message=\"error: Thread index 10 is out of range")

        # Test that an invalid --frame is handled
        # FIXME: no error is returned
        self.runCmd("-exec-step --frame 10")
        #self.expect("\^error: Frame index 10 is out of range")

    @lldbmi_test
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin due to calling convention assumptions")
    def test_lldbmi_exec_step_instruction(self):
        """Test that 'lldb-mi --interpreter' works for instruction stepping into."""

        self.spawnLldbMi(args = None)

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # Warning: the following is sensative to the lines in the
        # source and optimizations

        # Run to a_MyFunction call
        line = line_number('main.c', '//BP_a_MyFunction_call')
        self.runCmd("-break-insert -f main.c:%d" % line)
        self.expect("\^done,bkpt={number=\"1\"")
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Test that -exec-step-instruction steps over non branching
        # instruction
        self.runCmd("-exec-step-instruction --thread 1 --frame 0")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"end-stepping-range\".*main.c\",line=\"2[4-6]\"")

        # Test that -exec-step-instruction steps over non branching
        # instruction (and that --thread is optional)
        self.runCmd("-exec-step-instruction --frame 0")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"end-stepping-range\".*main.c\",line=\"2[4-6]\"")

        # Test that -exec-step-instruction steps into a_MyFunction
        # (and that --frame is optional)
        self.runCmd("-exec-step-instruction --thread 1")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"end-stepping-range\".*func=\"a_MyFunction\"")

        # Test that -exec-step-instruction steps into a_MyFunction
        # (and that both --thread and --frame are optional)
        self.runCmd("-exec-step-instruction --thread 1")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"end-stepping-range\".*func=\"a_MyFunction\"")

        # Test that an invalid --thread is handled
        self.runCmd("-exec-step-instruction --thread 0")
        self.expect("\^error,message=\"error: Thread index 0 is out of range")
        self.runCmd("-exec-step-instruction --thread 10")
        self.expect("\^error,message=\"error: Thread index 10 is out of range")

        # Test that an invalid --frame is handled
        # FIXME: no error is returned
        self.runCmd("-exec-step-instruction --frame 10")
        #self.expect("\^error: Frame index 10 is out of range")

    @lldbmi_test
    @expectedFailureWindows("llvm.org/pr22274: need a pexpect replacement for windows")
    @skipIfFreeBSD # llvm.org/pr22411: Failure presumably due to known thread races
    @skipIfLinux # llvm.org/pr22411: Failure presumably due to known thread races
    def test_lldbmi_exec_finish(self):
        """Test that 'lldb-mi --interpreter' works for -exec-finish."""

        self.spawnLldbMi(args = None)

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # Set argument 'l'
        self.runCmd("-exec-arguments l")
        self.expect("\^done")

        # Set BP at a_MyFunction_call and run to BP
        self.runCmd("-break-insert -f a_MyFunction")
        self.expect("\^done,bkpt={number=\"1\"")
        self.runCmd("-exec-run")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Test that -exec-finish returns from a_MyFunction
        self.runCmd("-exec-finish --thread 1 --frame 0")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"end-stepping-range\".*func=\"main\"")

        # Run to BP inside b_MyFunction call
        line = line_number('b.c', '//BP_b_MyFunction')
        self.runCmd("-break-insert -f b.c:%d" % line)
        self.expect("\^done,bkpt={number=\"2\"")
        self.runCmd("-exec-continue")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Test that -exec-finish hits BP at a_MyFunction call inside
        # b_MyFunction (and that --thread is optional)
        self.runCmd("-exec-finish --frame 0")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        # Test that -exec-finish returns from a_MyFunction call inside
        # b_MyFunction (and that --frame is optional)
        self.runCmd("-exec-finish --thread 1")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"end-stepping-range\".*func=\"b_MyFunction\"")

        # Test that -exec-finish returns from b_MyFunction
        # (and that both --thread and --frame are optional)
        self.runCmd("-exec-finish")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"end-stepping-range\".*func=\"main\"")

        # Test that an invalid --thread is handled
        self.runCmd("-exec-finish --thread 0")
        self.expect("\^error,message=\"error: Thread index 0 is out of range")
        self.runCmd("-exec-finish --thread 10")
        self.expect("\^error,message=\"error: Thread index 10 is out of range")

        # Test that an invalid --frame is handled
        # FIXME: no error is returned
        #self.runCmd("-exec-finish --frame 10")
        #self.expect("\^error: Frame index 10 is out of range")

        # Set BP at printf and run to BP
        # FIXME: BP at printf not resolved and never hit!
        self.runCmd("-interpreter-exec command \"b printf\"") #FIXME: self.runCmd("-break-insert -f printf")
        self.expect("\^done")                                 #FIXME: self.expect("\^done,bkpt={number=\"3\"")
        self.runCmd("-exec-continue")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"breakpoint-hit\"")

        ## Test that -exec-finish returns from printf
        self.runCmd("-exec-finish --thread 1 --frame 0")
        self.expect("\^running")
        self.expect("\*stopped,reason=\"end-stepping-range\".*func=\"main\"")

if __name__ == '__main__':
    unittest2.main()
