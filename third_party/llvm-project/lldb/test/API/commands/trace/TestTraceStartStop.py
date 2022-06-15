import lldb
from intelpt_testcase import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test.decorators import *

class TestTraceStartStop(TraceIntelPTTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)

    def expectGenericHelpMessageForStartCommand(self):
        self.expect("help thread trace start",
            substrs=["Syntax: thread trace start [<trace-options>]"])

    @testSBAPIAndCommands
    def testStartStopSessionFileThreads(self):
        # it should fail for processes from json session files
        self.expect("trace load -v " + os.path.join(self.getSourceDir(), "intelpt-trace", "trace.json"))

        # the help command should be the generic one, as it's not a live process
        self.expectGenericHelpMessageForStartCommand()

        self.traceStartThread(error=True)

        self.traceStopThread(error=True)

    @testSBAPIAndCommands
    def testStartWithNoProcess(self):
        self.traceStartThread(error=True)

    @testSBAPIAndCommands
    def testStartSessionWithWrongSize(self):
        self.expect("file " + os.path.join(self.getSourceDir(), "intelpt-trace", "a.out"))
        self.expect("b main")
        self.expect("r")

        self.traceStartThread(
            error=True, traceBufferSize=2000,
            substrs=["The trace buffer size must be a power of 2", "It was 2000"])

        self.traceStartThread(
            error=True, traceBufferSize=5000,
            substrs=["The trace buffer size must be a power of 2", "It was 5000"])

        self.traceStartThread(
            error=True, traceBufferSize=0,
            substrs=["The trace buffer size must be a power of 2", "It was 0"])

        self.traceStartThread(traceBufferSize=1048576)

    @skipIf(oslist=no_match(['linux']), archs=no_match(['i386', 'x86_64']))
    def testSBAPIHelp(self):
        self.expect("file " + os.path.join(self.getSourceDir(), "intelpt-trace", "a.out"))
        self.expect("b main")
        self.expect("r")

        help = self.getTraceOrCreate().GetStartConfigurationHelp()
        self.assertIn("traceBufferSize", help)
        self.assertIn("processBufferSizeLimit", help)

    @skipIf(oslist=no_match(['linux']), archs=no_match(['i386', 'x86_64']))
    def testStoppingAThread(self):
        self.expect("file " + os.path.join(self.getSourceDir(), "intelpt-trace", "a.out"))
        self.expect("b main")
        self.expect("r")
        self.expect("thread trace start")
        self.expect("n")
        self.expect("thread trace dump instructions", substrs=["""0x0000000000400511    movl   $0x0, -0x4(%rbp)
    no more data"""])
        # process stopping should stop the thread
        self.expect("process trace stop")
        self.expect("n")
        self.expect("thread trace dump instructions", substrs=["not traced"])


    @skipIf(oslist=no_match(['linux']), archs=no_match(['i386', 'x86_64']))
    def testStartStopLiveThreads(self):
        # The help command should be the generic one if there's no process running
        self.expectGenericHelpMessageForStartCommand()

        self.expect("thread trace start", error=True,
            substrs=["error: Process not available"])

        self.expect("file " + os.path.join(self.getSourceDir(), "intelpt-trace", "a.out"))
        self.expect("b main")

        self.expect("thread trace start", error=True,
            substrs=["error: Process not available"])

        # The help command should be the generic one if there's still no process running
        self.expectGenericHelpMessageForStartCommand()

        self.expect("r")

        # This fails because "trace start" hasn't been called yet
        self.expect("thread trace stop", error=True,
            substrs=["error: Process is not being traced"])


        # the help command should be the intel-pt one now
        self.expect("help thread trace start",
            substrs=["Start tracing one or more threads with intel-pt.",
                     "Syntax: thread trace start [<thread-index> <thread-index> ...] [<intel-pt-options>]"])

        # We start tracing with a small buffer size
        self.expect("thread trace start 1 --size 4096")

        # We fail if we try to trace again
        self.expect("thread trace start", error=True,
            substrs=["error: Thread ", "already traced"])

        # We can reconstruct the single instruction executed in the first line
        self.expect("n")
        self.expect("thread trace dump instructions -f",
            patterns=[f'''thread #1: tid = .*
  a.out`main \+ 4 at main.cpp:2
    0: {ADDRESS_REGEX}    movl'''])

        # We can reconstruct the instructions up to the second line
        self.expect("n")
        self.expect("thread trace dump instructions -f",
            patterns=[f'''thread #1: tid = .*
  a.out`main \+ 4 at main.cpp:2
    0: {ADDRESS_REGEX}    movl .*
  a.out`main \+ 11 at main.cpp:4
    1: {ADDRESS_REGEX}    movl .*
    2: {ADDRESS_REGEX}    jmp  .* ; <\+28> at main.cpp:4
    3: {ADDRESS_REGEX}    cmpl .*
    4: {ADDRESS_REGEX}    jle  .* ; <\+20> at main.cpp:5'''])

        self.expect("thread trace dump instructions",
            patterns=[f'''thread #1: tid = .*
  a.out`main \+ 32 at main.cpp:4
    4: {ADDRESS_REGEX}    jle  .* ; <\+20> at main.cpp:5
    3: {ADDRESS_REGEX}    cmpl .*
    2: {ADDRESS_REGEX}    jmp  .* ; <\+28> at main.cpp:4
    1: {ADDRESS_REGEX}    movl .*
  a.out`main \+ 4 at main.cpp:2
    0: {ADDRESS_REGEX}    movl .* '''])

        # We stop tracing
        self.expect("thread trace stop")

        # We can't stop twice
        self.expect("thread trace stop", error=True,
            substrs=["error: Thread ", "not currently traced"])

        # We trace again from scratch, this time letting LLDB to pick the current
        # thread
        self.expect("thread trace start")
        self.expect("n")
        self.expect("thread trace dump instructions -f",
            patterns=[f'''thread #1: tid = .*
  a.out`main \+ 20 at main.cpp:5
    0: {ADDRESS_REGEX}    xorl'''])

        self.expect("thread trace dump instructions",
            patterns=[f'''thread #1: tid = .*
  a.out`main \+ 20 at main.cpp:5
    0: {ADDRESS_REGEX}    xorl'''])

        self.expect("c")
        # Now the process has finished, so the commands should fail
        self.expect("thread trace start", error=True,
            substrs=["error: Process must be launched"])

        self.expect("thread trace stop", error=True,
            substrs=["error: Process must be launched"])

        # We should be able to trace the program if we relaunch it
        # For this, we'll trace starting at a different point in the new
        # process.
        self.expect("breakpoint disable")
        self.expect("b main.cpp:4")
        self.expect("r")
        self.expect("thread trace start")
        # We can reconstruct the single instruction executed in the first line
        self.expect("si")
        self.expect("thread trace dump instructions -c 1",
            patterns=[f'''thread #1: tid = .*
  a.out`main \+ 11 at main.cpp:4'''])
