"""
Test that debug symbols have the correct order as specified by the order file.
"""

import os, time
import re
import unittest2
import lldb
from lldbtest import *

class TestOrderFile(TestBase):

    mydir = "order"

    def test_order_file(self):
        """Test debug symbols follow the correct order by the order file."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Test that the debug symbols have Function f3 before Function f1.
        self.runCmd("image dump symtab a.out")
        output = self.res.GetOutput()
        mo_f3 = re.search("Function +.+f3", output)
        mo_f1 = re.search("Function +.+f1", output)
        
        # Match objects for f3 and f1 must exist and f3 must come before f1.
        self.assertTrue(mo_f3 and mo_f1 and mo_f3.start() < mo_f1.start(),
                        "Symbols have correct order by the order file")

        self.runCmd("run", RUN_COMPLETED)


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
