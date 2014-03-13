"""Test queues inspection SB APIs."""

import os, time
import unittest2
import lldb
import lldbutil
from lldbtest import *

class TestQueues(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test
    @dsym_test
    def test_with_dsym_and_python_api(self):
        """Test queues inspection SB APIs."""
        self.buildDsym()
        self.queues()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test
    @dwarf_test
    def test_with_dwarf_and_python_api(self):
        """Test queues inspection SB APIs."""
        self.buildDwarf()
        self.queues()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line numbers that we will step to in main:
        self.main_source = "main.c"

    def queues(self):
        """Test queues inspection SB APIs."""
        exe = os.path.join(os.getcwd(), "a.out")

        if not os.path.isfile('/Applications/Xcode.app/Contents/Developer/usr/lib/libBacktraceRecording.dylib'):
          self.skipTest ("Skipped because libBacktraceRecording.dylib was present on the system.")
          self.buildDefault()
          
        if not os.path.isfile('/usr/lib/system/introspection/libdispatch.dylib'):
          self.skipTest ("Skipped because introspection libdispatch dylib is not present.")
          self.buildDefault()
          
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        self.main_source_spec = lldb.SBFileSpec (self.main_source)

        break1 = target.BreakpointCreateByName ("stopper", 'a.out')
        self.assertTrue(break1, VALID_BREAKPOINT)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple (None, ['DYLD_INSERT_LIBRARIES=/Applications/Xcode.app/Contents/Developer/usr/lib/libBacktraceRecording.dylib', 'DYLD_LIBRARY_PATH=/usr/lib/system/introspection'], self.get_process_working_directory())

        self.assertTrue(process, PROCESS_IS_VALID)

        # The stop reason of the thread should be breakpoint.
        threads = lldbutil.get_threads_stopped_at_breakpoint (process, break1)
        if len(threads) != 1:
            self.fail ("Failed to stop at breakpoint 1.")

        libbtr_module_filespec = lldb.SBFileSpec()
        libbtr_module_filespec.SetFilename ("libBacktraceRecording.dylib")
        libbtr_module = target.FindModule (libbtr_module_filespec)
        if not libbtr_module.IsValid():
          self.skipTest ("Skipped because libBacktraceRecording.dylib was not loaded into the process.")
          self.buildDefault()

        self.assertTrue(process.GetNumQueues() == 4, "Found the correct number of queues.")

        q0 = process.GetQueueAtIndex(0)
        q1 = process.GetQueueAtIndex(1)
        q2 = process.GetQueueAtIndex(2)
        q3 = process.GetQueueAtIndex(3)

        self.assertTrue(q0.IsValid(), "queue 0 is valid")
        self.assertTrue(q1.IsValid(), "queue 1 is valid")
        self.assertTrue(q2.IsValid(), "queue 2 is valid")
        self.assertTrue(q3.IsValid(), "queue 3 is valid")

        self.assertTrue(q0.GetName() == "com.apple.work_submittor_1", "Get name of first queue")
        self.assertTrue(q1.GetName() == "com.apple.work_performer_1", "Get name of second queue")
        self.assertTrue(q2.GetName() == "com.apple.work_performer_2", "Get name of third queue")
        self.assertTrue(q3.GetName() == "com.apple.work_performer_3", "Get name of fourth queue")

        self.assertTrue(q0.GetQueueID() != 0, "Check queue 0 for valid QueueID")
        self.assertTrue(q1.GetQueueID() != 0, "Check queue 1 for valid QueueID")
        self.assertTrue(q2.GetQueueID() != 0, "Check queue 2 for valid QueueID")
        self.assertTrue(q3.GetQueueID() != 0, "Check queue 3 for valid QueueID")

        self.assertTrue(q0.GetNumPendingItems() == 0, "queue 0 should have no pending items")
        self.assertTrue(q0.GetNumRunningItems() == 1, "queue 0 should have one running item")

        self.assertTrue(q1.GetNumPendingItems() == 3, "queue 1 should have 3 pending items")
        self.assertTrue(q1.GetNumRunningItems() == 1, "queue 1 should have 1 running item")
    
        self.assertTrue(q2.GetNumPendingItems() == 9999, "queue 2 should have 9999 pending items")
        self.assertTrue(q2.GetNumRunningItems() == 1, "queue 2 should have 1 running item")

        self.assertTrue(q3.GetNumPendingItems() == 0, "queue 3 should have 0 pending items")
        self.assertTrue(q3.GetNumRunningItems() == 4, "queue 3 should have 4 running item")

        self.assertTrue(q0.GetNumThreads() == 1, "queue 0 should have 1 thread executing")
        self.assertTrue(q3.GetNumThreads() == 4, "queue 3 should have 4 threads executing")

        self.assertTrue(q0.GetKind() == lldb.eQueueKindSerial, "queue 0 is a serial queue")
        self.assertTrue(q1.GetKind() == lldb.eQueueKindSerial, "queue 1 is a serial queue")
        self.assertTrue(q2.GetKind() == lldb.eQueueKindSerial, "queue 2 is a serial queue")
        self.assertTrue(q3.GetKind() == lldb.eQueueKindConcurrent, "queue 3 is a concurrent queue")
        

        self.assertTrue(q1.GetThreadAtIndex(0).GetQueueID() == q1.GetQueueID(), "queue 1's thread should be owned by the same QueueID")
        self.assertTrue(q1.GetThreadAtIndex(0).GetQueueName() == q1.GetName(), "queue 1's thread should have the same queue name")

        self.assertTrue(q3.GetThreadAtIndex(0).GetQueueID() == q3.GetQueueID(), "queue 3's threads should be owned by the same QueueID")
        self.assertTrue(q3.GetThreadAtIndex(0).GetQueueName() == q3.GetName(), "queue 3's threads should have thes ame queue name")

        self.assertTrue(q2.GetPendingItemAtIndex(0).IsValid(), "queue 2's pending item #0 is valid")
        self.assertTrue(q2.GetPendingItemAtIndex(0).GetAddress().GetSymbol().GetName() == "doing_the_work_2", "queue 2's pending item #0 should be doing_the_work_2")
        self.assertTrue(q2.GetNumPendingItems() == 9999, "verify that queue 2 still has 9999 pending items")
        self.assertTrue(q2.GetPendingItemAtIndex(9998).IsValid(), "queue 2's pending item #9998 is valid")
        self.assertTrue(q2.GetPendingItemAtIndex(9998).GetAddress().GetSymbol().GetName() == "doing_the_work_2", "queue 2's pending item #0 should be doing_the_work_2")
        self.assertTrue(q2.GetPendingItemAtIndex(9999).IsValid() == False, "queue 2's pending item #9999 is invalid")

        
if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
