"""
Test lldb process IO launch flags..
"""

import os, time
import unittest2
import lldb
from lldbtest import *

class ProcessLaunchIOTestCase(TestBase):

    mydir = "process_io"

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_with_dsym (self):
        self.buildDsym ()
        self.process_io_test ()

    def test_with_dwarf (self):
        self.buildDwarf ()
        self.process_io_test ()

    def do_nothing (self):
        i = 1

    def process_io_test (self):
        exe = os.path.join (os.getcwd(), "a.out")
        self.expect("file " + exe,
                    patterns = [ "Current executable set to .*a.out" ])


        in_file = os.path.join (os.getcwd(), "input-file.txt")
        out_file = os.path.join (os.getcwd(), "output-test.out")
        err_file = os.path.join (os.getcwd(), "output-test.err")


        # Make sure the output files do not exist before launching the process
        try:
            os.remove (out_file)
        except OSError:
            # do_nothing (self)
            i = 1

        try:
            os.remove (err_file)
        except OSError:
            # do_nothing (self)
            i = 1

        launch_command = "process launch -i " + in_file + " -o " + out_file + " -e " + err_file
        
        self.expect (launch_command,
                     patterns = [ "Process .* launched: .*a.out" ])


        success = True
        err_msg = ""

        # Check to see if the 'stdout' file was created
        try:
            out_f = open (out_file)
        except IOError:
            success = False
            err_msg = err_msg + "   ERROR: stdout file was not created.\n"
        else:
            # Check to see if the 'stdout' file contains the right output
            line = out_f.readline ();
            if line != "This should go to stdout.\n":
                success = False
                err_msg = err_msg + "    ERROR: stdout file does not contain correct output.\n"
                out_f.close();
            
        # Try to delete the 'stdout' file
        try:
            os.remove (out_file)
        except OSError:
            # do_nothing (self)
            i = 1

        # Check to see if the 'stderr' file was created
        try:
            err_f = open (err_file)
        except IOError:
            success = False
            err_msg = err_msg + "     ERROR:  stderr file was not created.\n"
        else:
            # Check to see if the 'stderr' file contains the right output
            line = err_f.readline ()
            if line != "This should go to stderr.\n":
                success = False
                err_msg = err_msg + "    ERROR: stderr file does not contain correct output.\n\
"
                err_f.close()

        # Try to delete the 'stderr' file
        try:
            os.remove (err_file)
        except OSError:
            # do_nothing (self)
            i = 1

        if not success:
            # This test failed, but we need to make the main testing
            # mechanism realize something is wrong.
            #
            # First, print out the real error message.
            self.fail (err_msg)
            #print err_msg

            # Second, force a test case to fail:
            #self.expect ("help quit",
            #             patterns = ["Intentional failure .*"])

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()

