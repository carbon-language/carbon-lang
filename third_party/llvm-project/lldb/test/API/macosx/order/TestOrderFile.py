"""
Test that debug symbols have the correct order as specified by the order file.
"""



import re
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class OrderFileTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipUnlessDarwin
    def test(self):
        """Test debug symbols follow the correct order by the order file."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Test that the debug symbols have Function f3 before Function f1.
        # Use "-s address" option to sort by address.
        self.runCmd("image dump symtab -s address %s" % exe)
        output = self.res.GetOutput()
        mo_f3 = re.search("Code +.+f3", output)
        mo_f1 = re.search("Code +.+f1", output)

        # Match objects for f3 and f1 must exist and f3 must come before f1.
        self.assertTrue(mo_f3 and mo_f1 and mo_f3.start() < mo_f1.start(),
                        "Symbols have correct order by the order file")

        self.runCmd("run", RUN_COMPLETED)
