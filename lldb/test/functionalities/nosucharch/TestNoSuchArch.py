"""
Test that using a non-existent architecture name does not crash LLDB.
"""
import lldb
import unittest2
from lldbtest import *
import lldbutil

class NoSuchArchTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

        
    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_with_dsym (self):
        self.buildDsym()
        self.do_test ()


    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dwarf_test
    def test_with_dwarf (self):
        self.buildDwarf()
        self.do_test ()

    def do_test (self):
        exe = os.path.join (os.getcwd(), "a.out")
        
        # Check that passing an invalid arch via the command-line fails but doesn't crash
        self.expect("target crete --arch nothingtoseehere %s" % (exe), error=True)
        
        
        # Check that passing an invalid arch via the SB API fails but doesn't crash
        target = self.dbg.CreateTargetWithFileAndArch(exe,"nothingtoseehere")
        
        self.assertFalse(target.IsValid(), "This target should not be valid")
        
        # Now just create the target with the default arch and check it's fine
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target.IsValid(), "This target should now be valid")

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()

