"""
Test that clang produces the __apple accelerator tables, for example, __apple_types, correctly.
"""

import os, time
import unittest2
import lldb
from lldbtest import *
from lldbutil import symbol_type_to_str

class AppleTypesTestCase(TestBase):

    mydir = os.path.join("macosx", "debug-info", "apple_types")

    #rdar://problem/11166975
    @unittest2.expectedFailure
    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_debug_info_for_apple_types(self):
        """Test that __apple_types section does get produced by clang."""

        if not self.getCompiler().endswith('clang'):
            self.skipTest("clang compiler only test")

        self.buildDefault()
        self.apple_types()

    def apple_types(self):
        """Test that __apple_types section does get produced by clang."""
        exe = os.path.join(os.getcwd(), "main.o")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)
        self.assertTrue(target.GetNumModules() > 0)

        # Hide stdout if not running with '-t' option.
        if not self.TraceOn():
            self.HideStdout()

        print "Number of modules for the target: %d" % target.GetNumModules()
        for module in target.module_iter():
            print module

        # Get the executable module at index 0.
        exe_module = target.GetModuleAtIndex(0)

        debug_str_section = exe_module.FindSection("__debug_str")
        self.assertTrue(debug_str_section)
        print "__debug_str section:", debug_str_section

        # Find our __apple_types section by name.
        apple_types_section = exe_module.FindSection("__apple_types")
        self.assertTrue(apple_types_section)
        print "__apple_types section:", apple_types_section


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
