"""
Test lldb-mi =library-loaded notifications.
"""

import lldbmi_testcase
from lldbtest import *
import unittest2

class MiLibraryLoadedTestCase(lldbmi_testcase.MiTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)

    @lldbmi_test
    @skipIfWindows #llvm.org/pr24452: Get lldb-mi tests working on Windows
    @skipIfFreeBSD # llvm.org/pr22411: Failure presumably due to known thread races
    def test_lldbmi_library_loaded(self):
        """Test that 'lldb-mi --interpreter' shows the =library-loaded notifications."""

        self.spawnLldbMi(args = None)

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # Test =library-loaded
        import os
        path = os.path.join(os.getcwd(), self.myexe)
        symbols_path = os.path.join(path + ".dSYM", "Contents", "Resources", "DWARF", self.myexe)
        def add_slashes(x): return x.replace("\\", "\\\\").replace("\"", "\\\"").replace("\'", "\\\'").replace("\0", "\\\0")
        self.expect([ "=library-loaded,id=\"%s\",target-name=\"%s\",host-name=\"%s\",symbols-loaded=\"1\",symbols-path=\"%s\",loaded_addr=\"-\",size=\"[0-9]+\"" % (add_slashes(path), add_slashes(path), add_slashes(path), add_slashes(symbols_path)),
                      "=library-loaded,id=\"%s\",target-name=\"%s\",host-name=\"%s\",symbols-loaded=\"0\",loaded_addr=\"-\",size=\"[0-9]+\"" % (add_slashes(path), add_slashes(path), add_slashes(path)) ])

if __name__ == '__main__':
    unittest2.main()
