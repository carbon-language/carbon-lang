"""
Test lldb logging.  This test just makes sure logging doesn't crash, and produces some output.
"""

import os, time
import unittest2
import lldb
from lldbtest import *

class LogTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_with_dsym (self):
        self.buildDsym ()
        self.command_log_tests ("dsym")

    @dwarf_test
    def test_with_dwarf (self):
        self.buildDwarf ()
        self.command_log_tests ("dwarf")

    def command_log_tests (self, type):
        exe = os.path.join (os.getcwd(), "a.out")
        self.expect("file " + exe,
                    patterns = [ "Current executable set to .*a.out" ])

        log_file = os.path.join (os.getcwd(), "lldb-commands-log-%s-%s-%s.txt" % (type,
                                                                                  os.path.basename(self.getCompiler()),
                                                                                  self.getArchitecture()))

        if (os.path.exists (log_file)):
            os.remove (log_file)

        # By default, Debugger::EnableLog() will set log options to
        # PREPEND_THREAD_NAME + OPTION_THREADSAFE. We don't want the
        # threadnames here, so we enable just threadsafe (-t).
        self.runCmd ("log enable -t -f '%s' lldb commands" % (log_file))
        
        self.runCmd ("command alias bp breakpoint")
                     
        self.runCmd ("bp set -n main")

        self.runCmd ("bp l")

        self.runCmd("log disable lldb")

        self.assertTrue (os.path.isfile (log_file))

        f = open (log_file)
        log_lines = f.readlines()
        f.close ()
        os.remove (log_file)

        self.assertTrue(log_lines > 0, "Something was written to the log file.")

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()

