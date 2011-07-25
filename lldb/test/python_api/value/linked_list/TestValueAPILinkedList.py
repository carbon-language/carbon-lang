"""
Test SBValue API linked_list_iter which treats the SBValue as a linked list and
supports iteration till the end of list is reached.
"""

import os, time
import re
import unittest2
import lldb, lldbutil
from lldbtest import *

class ValueAsLinkedListTestCase(TestBase):

    mydir = os.path.join("python_api", "value", "linked_list")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test
    def test_with_dsym(self):
        """Exercise SBValue API linked_list_iter."""
        d = {'EXE': self.exe_name}
        self.buildDsym(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.linked_list_api(self.exe_name)

    @python_api_test
    def test_with_dwarf(self):
        """Exercise SBValue API linked_list_iter."""
        d = {'EXE': self.exe_name}
        self.buildDwarf(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.linked_list_api(self.exe_name)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # We'll use the test method name as the exe_name.
        self.exe_name = self.testMethodName
        # Find the line number to break at.
        self.line = line_number('main.cpp', '// Break at this line')

    def linked_list_api(self, exe_name):
        """Exercise SBValue API linked_list-iter."""
        exe = os.path.join(os.getcwd(), exe_name)

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Create the breakpoint inside function 'main'.
        breakpoint = target.BreakpointCreateByLocation('main.cpp', self.line)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(None, None, os.getcwd())
        self.assertTrue(process, PROCESS_IS_VALID)

        def eol(val):
            """Test to determine end of list."""
            if not val:
                return True
            try:
                id = val.GetChildMemberWithName("id")
                if int(id.GetValue()) > 0:
                    return False
            except:
                #exc_type, exc_value, exc_tb = sys.exc_info()
                #traceback.print_exception(exc_type, exc_value, exc_tb)
                pass

            # If we fall through to here.  It could be that exception
            # occurred or the "id" child member does not qualify as a
            # valid item. Return True for EOL.
            return True

        # Get Frame #0.
        self.assertTrue(process.GetState() == lldb.eStateStopped)
        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertTrue(thread != None, "There should be a thread stopped due to breakpoint condition")
        frame0 = thread.GetFrameAtIndex(0)

        # Get variable 'task_head'.
        task_head = frame0.FindVariable('task_head')
        self.assertTrue(task_head, VALID_VARIABLE)
        self.DebugSBValue(task_head)

        # By design (see main.cpp), the visited id's are: [1, 2, 4, 5].
        visitedIDs = [1, 2, 4, 5]
        list = []

        cvf = lldbutil.ChildVisitingFormatter(indent_child=2)
        for t in task_head.linked_list_iter('next', eol):
            self.assertTrue(t, VALID_VARIABLE)
            list.append(int(t.GetChildMemberWithName("id").GetValue()))
            if self.TraceOn():
                print cvf.format(t)

        # Sanity checks that the we visited all the items (no more, no less).
        #print "list:", list
        self.assertTrue(visitedIDs == list)
        
if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
