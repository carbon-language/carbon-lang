"""
Test lldb Python commands.
"""

import os, time
import unittest2
import lldb
from lldbtest import *

class CmdPythonTestCase(TestBase):

    mydir = os.path.join("functionalities", "command_python")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_with_dsym (self):
        self.buildDsym ()
        self.pycmd_tests ()

    def test_with_dwarf (self):
        self.buildDwarf ()
        self.pycmd_tests ()

    def pycmd_tests (self):
        exe = os.path.join (os.getcwd(), "a.out")
        self.expect("file " + exe,
                    patterns = [ "Current executable set to .*a.out" ])

        self.runCmd("command source py_import")

        # We don't want to display the stdout if not in TraceOn() mode.
        if not self.TraceOn():
            self.HideStdout()

        self.expect('welcome Enrico',
            substrs = ['Hello Enrico, welcome to LLDB']);
                
        self.expect("help welcome",
                    substrs = ['Just a docstring for welcome_impl',
                               'A command that says hello to LLDB users'])

        self.runCmd("command script delete welcome");

        self.expect('welcome Enrico', matching=False, error=True,
                substrs = ['Hello Enrico, welcome to LLDB']);

        self.expect('targetname',
            substrs = ['a.out'])

        self.expect('targetname fail', error=True,
                    substrs = ['a test for error in command'])

        self.expect('command script list',
            substrs = ['targetname',
                       'Run Python function welcome.target_name_impl'])

        self.expect("help targetname",
                    substrs = ['Run Python function welcome.target_name_imp',
                               'This command takes \'raw\' input',
                               'quote stuff'])

        self.expect("longwait",
                    substrs = ['Done; if you saw the delays I am doing OK'])

        self.runCmd("command script clear")

        self.expect('command script list', matching=False,
                    substrs = ['targetname',
                               'longwait'])

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()

