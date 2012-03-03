"""
Test lldb data formatter subsystem.
"""

import os, time
import unittest2
import lldb
from lldbtest import *

class PythonSynthDataFormatterTestCase(TestBase):

    mydir = os.path.join("functionalities", "data-formatter", "data-formatter-python-synth")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_with_dsym_and_run_command(self):
        """Test data formatter commands."""
        self.buildDsym()
        self.data_formatter_commands()

    def test_with_dwarf_and_run_command(self):
        """Test data formatter commands."""
        self.buildDwarf()
        self.data_formatter_commands()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number('main.cpp', '// Set break point at this line.')

    def data_formatter_commands(self):
        """Test that that file and class static variables display correctly."""
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        self.expect("breakpoint set -f main.cpp -l %d" % self.line,
                    BREAKPOINT_CREATED,
            startstr = "Breakpoint created: 1: file ='main.cpp', line = %d, locations = 1" %
                        self.line)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd('type format clear', check=False)
            self.runCmd('type summary clear', check=False)
            self.runCmd('type filter clear', check=False)
            self.runCmd('type synth clear', check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        # print the f00_1 variable without a synth
        self.expect("frame variable f00_1",
            substrs = ['a = 0',
                       'b = 1',
                       'r = 33']);

        # now set up the synth
        self.runCmd("script from fooSynthProvider import *")
        self.runCmd("type synth add -l fooSynthProvider foo")

        # check that we get the two real vars and the fake_a variables
        self.expect("frame variable f00_1",
                    substrs = ['r = 33',
                               'fake_a = 16777216',
                               'a = 0']);

        # check that we do not get the extra vars
        self.expect("frame variable f00_1", matching=False,
                    substrs = ['b = 1']);
        
        # check access to members by name
        self.expect('frame variable f00_1.fake_a',
                substrs = ['16777216'])
        
        # check access to members by index
        self.expect('frame variable f00_1[1]',
                    substrs = ['16777216'])
        
        # put synthetic children in summary in several combinations
        self.runCmd("type summary add --summary-string \"fake_a=${svar.fake_a}\" foo")
        self.expect('frame variable f00_1',
                    substrs = ['fake_a=16777216'])
        self.runCmd("type summary add --summary-string \"fake_a=${svar[1]}\" foo")
        self.expect('frame variable f00_1',
            substrs = ['fake_a=16777216'])
        
        # clear the summary
        self.runCmd("type summary delete foo")

        # check that the caching does not span beyond the stopoint
        self.runCmd("n")

        self.expect("frame variable f00_1",
                    substrs = ['r = 33',
                               'fake_a = 16777216',
                               'a = 1']);

        # check that altering the object also alters fake_a
        self.runCmd("expr f00_1.a = 280")
        self.expect("frame variable f00_1",
                    substrs = ['r = 33',
                               'fake_a = 16777217',
                               'a = 280']);

        # check that expanding a pointer does the right thing
        self.expect("frame variable -P 1 f00_ptr",
            substrs = ['r = 45',
                       'fake_a = 218103808',
                       'a = 12'])
        
        # now add a filter.. it should fail
        self.expect("type filter add foo --child b --child j", error=True,
                substrs = ['cannot add'])
        
        # we get the synth again..
        self.expect('frame variable f00_1', matching=False,
            substrs = ['b = 1',
                       'j = 17'])
        self.expect("frame variable -P 1 f00_ptr",
                    substrs = ['r = 45',
                               'fake_a = 218103808',
                               'a = 12'])
        
        # now delete the synth and add the filter
        self.runCmd("type synth delete foo")
        self.runCmd("type filter add foo --child b --child j")
        
        self.expect('frame variable f00_1',
                        substrs = ['b = 1',
                                   'j = 17'])
        self.expect("frame variable -P 1 f00_ptr", matching=False,
                    substrs = ['r = 45',
                               'fake_a = 218103808',
                               'a = 12'])
        
        # now add the synth and it should fail
        self.expect("type synth add -l fooSynthProvider foo", error=True,
                    substrs = ['cannot add'])
        
        # check the listing
        self.expect('type synth list', matching=False,
                    substrs = ['foo',
                               'Python class fooSynthProvider'])
        self.expect('type filter list', 
                    substrs = ['foo',
                               '.b',
                               '.j'])
        
        # delete the filter, add the synth
        self.runCmd("type filter delete foo")
        self.runCmd("type synth add -l fooSynthProvider foo")
        
        self.expect('frame variable f00_1', matching=False,
                    substrs = ['b = 1',
                               'j = 17'])
        self.expect("frame variable -P 1 f00_ptr", 
                    substrs = ['r = 45',
                               'fake_a = 218103808',
                               'a = 12'])

        # check the listing
        self.expect('type synth list',
                    substrs = ['foo',
                               'Python class fooSynthProvider'])
        self.expect('type filter list', matching=False,
                    substrs = ['foo',
                               '.b',
                               '.j'])
        
        # delete the synth and check that we get good output
        self.runCmd("type synth delete foo")
        
        self.expect("frame variable f00_1",
                    substrs = ['a = 280',
                               'b = 1',
                               'j = 17']);

        self.expect("frame variable f00_1", matching=False,
                substrs = ['fake_a = '])
        
        self.runCmd("n")
        
        self.runCmd("script from ftsp import *")
        self.runCmd("type synth add -l ftsp wrapint")
        
        self.expect('frame variable test_cast',
            substrs = ['A',
                       'B',
                       'C',
                       'D'])

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
