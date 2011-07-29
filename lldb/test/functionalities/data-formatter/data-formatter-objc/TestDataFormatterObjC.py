"""
Test lldb data formatter subsystem.
"""

import os, time
import unittest2
import lldb
from lldbtest import *

class DataFormatterTestCase(TestBase):

    mydir = os.path.join("functionalities", "data-formatter", "data-formatter-objc")

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
        self.line = line_number('main.m', '// Set break point at this line.')

    def data_formatter_commands(self):
        """Test that that file and class static variables display correctly."""
        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        self.expect("breakpoint set -f main.m -l %d" % self.line,
                    BREAKPOINT_CREATED,
            startstr = "Breakpoint created: 1: file ='main.m', line = %d, locations = 1" %
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
            self.runCmd('type synth clear', check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        self.runCmd("type summary add -f \"${var%@}\" MyClass")

        self.expect("frame variable object2",
            substrs = ['MyOtherClass']);
        
        self.expect("frame variable *object2",
            substrs = ['MyOtherClass']);

        # Now let's delete the 'MyClass' custom summary.
        self.runCmd("type summary delete MyClass")

        # The type format list should not show 'MyClass' at this point.
        self.expect("type summary list", matching=False,
            substrs = ['MyClass'])

        self.runCmd("type summary add -f \"a test\" MyClass")
        
        self.expect("frame variable object2",
                    substrs = ['a test']);
        
        self.expect("frame variable *object2",
                    substrs = ['a test']);

        self.expect("frame variable object",
                    substrs = ['a test']);
        
        self.expect("frame variable *object",
                    substrs = ['a test']);

        self.runCmd("type summary add -f \"a test\" MyClass -C no")
        
        self.expect("frame variable *object2",
                    substrs = ['*object2 = {',
                               'MyClass = a test',
                               'backup = ']);
        
        self.expect("frame variable object2", matching=False,
                    substrs = ['a test']);
        
        self.expect("frame variable object",
                    substrs = ['a test']);
        
        self.expect("frame variable *object",
                    substrs = ['a test']);

        # Now check that the synth for CFString works
        self.runCmd("script from CFString import *")
        self.runCmd("type synth add -l CFStringSynthProvider NSString")

        self.expect('frame variable str -P 1 -Y',
            substrs = ['mutable =',
                       'inline = ',
                       'explicit = ',
                       'content = ',
                       'A rather short ASCII NSString object is here'])

        self.expect('frame variable str2 -P 1 -Y',
                    substrs = ['mutable =',
                               'inline = ',
                               'explicit = ',
                               'content = ',
                               'A rather short UTF8 NSString object is here'])

        self.expect('frame variable str3 -P 1 -Y',
                    substrs = ['mutable =',
                               'inline = ',
                               'explicit = ',
                               'content = ',
                               'A string made with the at sign is here'])

        self.expect('frame variable str4 -P 1 -Y',
                    substrs = ['mutable =',
                               'inline = ',
                               'explicit = ',
                               'content = ',
                               'This is string number 4 right here'])
        
        self.expect('frame variable str5 -P 1 -Y',
                    substrs = ['mutable =',
                               'inline = ',
                               'explicit = ',
                               'content = ',
                               '{{1, 1}, {5, 5}}'])
                
        self.expect('frame variable str6 -P 1 -Y',
                    substrs = ['mutable =',
                               'inline = ',
                               'explicit = ',
                               'content = ',
                               '1ST'])
        
        self.expect('frame variable str7 -P 1 -Y',
                    substrs = ['mutable =',
                               'inline = ',
                               'explicit = ',
                               'content = ',
                               '\\xcf\\x83xx'])
        
        self.expect('frame variable str8 -P 1 -Y',
                    substrs = ['mutable =',
                               'inline = ',
                               'explicit = ',
                               'content = ',
                               'hasVeryLongExtensionThisTime'])

        self.expect('frame variable str9 -P 1 -Y',
                    substrs = ['mutable =',
                               'inline = ',
                               'explicit = ',
                               'content = ',
                               'a very much boring task to write a string this way!!\\xe4\\x8c\\xb3'])
        
        self.expect('frame variable str10 -P 1 -Y',
                    substrs = ['mutable =',
                               'inline = ',
                               'explicit = ',
                               'content = ',
                               'This is a Unicode string \\xcf\\x83 number 4 right here'])
        
        self.expect('frame variable str11 -P 1 -Y',
                    substrs = ['mutable =',
                               'inline = ',
                               'explicit = ',
                               'content = ',
                               '__NSCFString'])
        
        self.expect('frame variable processName -P 1 -Y',
                    substrs = ['mutable =',
                               'inline = ',
                               'explicit = ',
                               'content = ',
                               'a.out'])
        
        self.expect('frame variable str12 -P 1 -Y',
                    substrs = ['mutable =',
                               'inline = ',
                               'explicit = ',
                               'content = ',
                               'Process Name:  a.out Process Id:'])
        
        # delete the synth and set a summary
        self.runCmd("type synth delete NSString")
        self.runCmd("type summary add -F CFString_SummaryProvider NSString")

        self.expect('frame variable str',
            substrs = ['A rather short ASCII NSString object is here'])
        self.expect('frame variable str2',
                    substrs = ['A rather short UTF8 NSString object is here'])
        self.expect('frame variable str3',
                    substrs = ['A string made with the at sign is here'])
        self.expect('frame variable str4',
                    substrs = ['This is string number 4 right here'])
        self.expect('frame variable str5',
                    substrs = ['{{1, 1}, {5, 5}}'])
        self.expect('frame variable str6',
                    substrs = ['1ST'])
        self.expect('frame variable str7',
                    substrs = ['\\xcf\\x83xx'])
        self.expect('frame variable str8',
                    substrs = ['hasVeryLongExtensionThisTime'])
        self.expect('frame variable str9',
                    substrs = ['a very much boring task to write a string this way!!\\xe4\\x8c\\xb3'])
        self.expect('frame variable str10',
                    substrs = ['This is a Unicode string \\xcf\\x83 number 4 right here'])
        self.expect('frame variable str11',
                    substrs = ['__NSCFString'])
        self.expect('frame variable processName',
                    substrs = ['a.out'])        
        self.expect('frame variable str12',
                    substrs = ['Process Name:  a.out Process Id:'])

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
