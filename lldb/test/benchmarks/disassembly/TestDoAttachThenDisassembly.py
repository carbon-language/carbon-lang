"""Test lldb's disassemblt speed."""

import os, sys
import unittest2
import lldb
import pexpect
from lldbbench import *

class AttachThenDisassemblyBench(BenchBase):

    mydir = os.path.join("benchmarks", "disassembly")

    def setUp(self):
        BenchBase.setUp(self)

    @benchmarks_test
    def test_attach_then_disassembly(self):
        """Attach to a spawned lldb process then run disassembly benchmarks."""
        print
        self.run_lldb_attach_then_disassembly(10)
        print "lldb disassembly benchmark:", self.stopwatch

    def run_lldb_attach_then_disassembly(self, count):
        target = self.dbg.CreateTarget(self.lldbHere)

        # Spawn a new process and don't display the stdout if not in TraceOn() mode.
        import subprocess
        popen = subprocess.Popen([self.lldbHere, self.lldbOption],
                                 stdout = open(os.devnull, 'w') if not self.TraceOn() else None)
        if self.TraceOn():
            print "pid of spawned process: %d" % popen.pid

        # Attach to the launched lldb process.
        listener = lldb.SBListener("my.attach.listener")
        error = lldb.SBError()
        process = target.AttachToProcessWithID(listener, popen.pid, error)

        # Set thread0 as the selected thread, followed by the 'MainLoop' frame
        # as the selected frame.  Then do disassembly on the function.
        thread0 = process.GetThreadAtIndex(0)
        process.SetSelectedThread(thread0)
        i = 0
        found = False
        for f in thread0:
            #print "frame#%d %s" % (i, f.GetFunctionName())
            if "MainLoop" in f.GetFunctionName():
                found = True
                thread0.SetSelectedFrame(i)
                if self.TraceOn():
                    print "Found frame#%d for function 'MainLoop'" % i
                break
            i += 1
            
        # Reset the stopwatch now.
        self.stopwatch.reset()
        for i in range(count):
            with self.stopwatch:
                # Disassemble the function.
                self.runCmd("disassemble -f")


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
