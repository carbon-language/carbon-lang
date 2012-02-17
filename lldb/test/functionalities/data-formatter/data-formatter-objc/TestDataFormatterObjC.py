"""
Test lldb data formatter subsystem.
"""

import os, time
import unittest2
import lldb
from lldbtest import *

class ObjCDataFormatterTestCase(TestBase):

    mydir = os.path.join("functionalities", "data-formatter", "data-formatter-objc")

    # rdar://problem/10153585 lldb ToT regression of test suite with r139772 check-in
    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_with_dsym_and_run_command(self):
        """Test data formatter commands."""
        self.buildDsym()
        self.data_formatter_commands()

    # rdar://problem/10153585 lldb ToT regression of test suite with r139772 check-in
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
            self.runCmd('type category disable CoreFoundation', check=False)
            self.runCmd('type category disable CoreGraphics', check=False)
            self.runCmd('type category disable CoreServices', check=False)


        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        self.runCmd("type summary add --summary-string \"${var%@}\" MyClass")

        self.expect("frame variable object2",
            substrs = ['MyOtherClass']);
        
        self.expect("frame variable *object2",
            substrs = ['MyOtherClass']);

        # Now let's delete the 'MyClass' custom summary.
        self.runCmd("type summary delete MyClass")

        # The type format list should not show 'MyClass' at this point.
        self.expect("type summary list", matching=False,
            substrs = ['MyClass'])

        self.runCmd("type summary add --summary-string \"a test\" MyClass")
        
        self.expect("frame variable object2",
                    substrs = ['a test']);
        
        self.expect("frame variable *object2",
                    substrs = ['a test']);

        self.expect("frame variable object",
                    substrs = ['a test']);
        
        self.expect("frame variable *object",
                    substrs = ['a test']);

        self.runCmd("type summary add --summary-string \"a test\" MyClass -C no")
        
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
                               'a very much boring task to write a string this way!!\\xcf\\x83'])
        
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
                               'NSCFString'])
        
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
        
        # check that access to synthetic children by name works
        self.expect("frame variable str12->mutable",
            substrs = ['(int) mutable = 0'])
        
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
                    substrs = ['a very much boring task to write a string this way!!\\xcf\\x83'])
        self.expect('frame variable str10',
                    substrs = ['This is a Unicode string \\xcf\\x83 number 4 right here'])
        self.expect('frame variable str11',
                    substrs = ['NSCFString'])
        self.expect('frame variable processName',
                    substrs = ['a.out'])        
        self.expect('frame variable str12',
                    substrs = ['Process Name:  a.out Process Id:'])
        self.expect('frame variable dyn_test', matching=False,
                    substrs = ['Process Name:  a.out Process Id:'])
        self.expect('frame variable dyn_test -d run-target -T',
                    substrs = ['(__NSCFString *, dynamic type:',
                               'Process Name:  a.out Process Id:'])
        self.expect('frame variable dyn_test -d run-target',
                    substrs = ['(__NSCFString *)',
                               'Process Name:  a.out Process Id:'])

            
        # check that we can format stuff out of the expression parser
        self.expect('expression ((id)@"Hello")', matching=False,
                    substrs = ['Hello'])
            
        self.expect('expression -d true -- ((id)@"Hello")',
        substrs = ['Hello'])

        self.expect('expr -d true -- label1',
            substrs = ['Process Name'])

        self.expect('expr -d true -- @"Hello"',
            substrs = ['Hello'])

        # check formatters for common Objective-C types
        self.runCmd('type category enable CoreFoundation')
        self.runCmd('type category enable CoreGraphics')
        self.runCmd('type category enable CoreServices')
        self.expect("frame variable",
             substrs = ['(CFGregorianUnits) cf_greg_units = 1 years, 3 months, 5 days, 12 hours, 5 minutes 7 seconds',
             '(CFRange) cf_range = location=4 length=4',
             '(NSPoint) ns_point = x=4, y=4',
             '(NSRange) ns_range = location=4, length=4',
             '(NSRect *) ns_rect_ptr = (x=1, y=1), (width=5, height=5)',
             '(NSRect) ns_rect = (x=1, y=1), (width=5, height=5)',
             '(NSRectArray) ns_rect_arr = ((x=1, y=1), (width=5, height=5)), ...',
             '(NSSize) ns_size = width=5, height=7',
             '(NSSize *) ns_size_ptr = width=5, height=7',
             '(CGSize) cg_size = (width=1, height=6)',
             '(CGPoint) cg_point = (x=2, y=7)',
             '(CGRect) cg_rect = origin=(x=1, y=2) size=(width=7, height=7)',
             '(RGBColor) rgb_color = red=3 green=56 blue=35',
             '(RGBColor *) rgb_color_ptr = red=3 green=56 blue=35',
             '(Rect) rect = (t=4, l=8, b=4, r=7)',
             '(Rect *) rect_ptr = (t=4, l=8, b=4, r=7)',
             '(Point) point = (v=7, h=12)',
             '(Point *) point_ptr = (v=7, h=12)',
             '(HIPoint) hi_point = (x=7, y=12)',
             '(HIRect) hi_rect = origin=(x=3, y=5) size=(width=4, height=6)'])

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
