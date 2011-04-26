"""
Test lldbutil.lldb_iter() which returns an iterator object for lldb's aggregate
data structures.
"""

import os, time
import re
import unittest2
import lldb
from lldbtest import *

class LLDBIteratorTestCase(TestBase):

    mydir = "python_api/lldbutil"

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number('main.cpp', '// Set break point at this line.')

    def test_lldb_iter(self):
        """Test lldb_iter works correctly."""
        self.buildDefault()
        self.lldb_iter_test()

    def lldb_iter_test(self):
        exe = os.path.join(os.getcwd(), "a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target.IsValid(), VALID_TARGET)

        breakpoint = target.BreakpointCreateByLocation("main.cpp", self.line)
        self.assertTrue(breakpoint.IsValid(), VALID_BREAKPOINT)

        # Now launch the process, and do not stop at entry point.
        rc = lldb.SBError()
        self.process = target.Launch (self.dbg.GetListener(), None, None, os.ctermid(), os.ctermid(), os.ctermid(), None, 0, False, rc)

        if not rc.Success() or not self.process.IsValid():
            self.fail("SBTarget.LaunchProcess() failed")

        from lldbutil import lldb_iter, get_description
        yours = []
        for i in range(target.GetNumModules()):
            yours.append(target.GetModuleAtIndex(i))
        mine = []
        for m in lldb_iter(target, 'GetNumModules', 'GetModuleAtIndex'):
            mine.append(m)

        self.assertTrue(len(yours) == len(mine))
        for i in range(len(yours)):
            if self.TraceOn():
                print "yours[%d]='%s'" % (i, get_description(yours[i]))
                print "mine[%d]='%s'" % (i, get_description(mine[i]))
            self.assertTrue(yours[i].GetUUIDString() == mine[i].GetUUIDString(),
                            "UUID of yours[%d] and mine[%d] matches" % (i, i))


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
