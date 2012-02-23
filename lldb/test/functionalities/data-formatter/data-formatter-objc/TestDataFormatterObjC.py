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
            self.runCmd('type category disable AppKit', check=False)


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

        # Now enable AppKit and check we are displaying Cocoa classes correctly
        self.runCmd("type category enable AppKit")
        self.expect('frame variable num1 num2 num3 num4 num5 num6 num7 num8_Y num8_N num9',
                    substrs = ['(NSNumber *) num1 = 0x0000000000000583 (int)5',
                    '(NSNumber *) num2 = ',' (float)3.1',
                    '(NSNumber *) num3 = ',' (double)3.14',
                    '(NSNumber *) num4 = ',' (long)18446744073709551614',
                    '(NSNumber *) num5 = ',' (char)65',
                    '(NSNumber *) num6 = ',' (long)255',
                    '(NSNumber *) num7 = ',' (long)2000000',
                    '(NSNumber *) num8_Y = ',' @"1"',
                    '(NSNumber *) num8_N = ',' @"0"',
                    '(NSNumber *) num9 = ',' (short)33920'])

        self.expect('frame variable str0 str1 str2 str3 str4 str5 str6 str8 str9 str10 str11 label1 label2 processName str12',
                    substrs = ['(NSString *) str1 = ',' @"A rather short ASCII NSString object is here"',
                    '(NSString *) str0 = ',' @"255"',
                    '(NSString *) str1 = ',' @"A rather short ASCII NSString object is here"',
                    '(NSString *) str2 = ',' @"A rather short UTF8 NSString object is here"',
                    '(NSString *) str3 = ',' @"A string made with the at sign is here"',
                    '(NSString *) str4 = ',' @"This is string number 4 right here"',
                    '(NSString *) str5 = ',' @"{{1, 1}, {5, 5}}"',
                    '(NSString *) str6 = ',' @"1ST"',
                    '(NSString *) str8 = ',' @"hasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTime',
                    '(NSString *) str9 = ',' @"a very much boring task to write a string this way!!',
                    '(NSString *) str10 = ',' @"This is a Unicode string',
                    '(NSString *) str11 = ',' @"__NSCFString"',
                    '(NSString *) label1 = ',' @"Process Name: "',
                    '(NSString *) label2 = ',' @"Process Id: "',
                    '(NSString *) str12 = ',' @"Process Name:  a.out Process Id:'])

        self.expect('frame variable newArray newDictionary newMutableDictionary cfdict_ref mutable_dict_ref cfarray_ref mutable_array_ref',
                    substrs = ['(NSArray *) newArray = ',' size=50',
                    '(NSDictionary *) newDictionary = ',' 12 key/value pairs',
                    '(NSDictionary *) newMutableDictionary = ',' 21 key/value pairs',
                    '(CFDictionaryRef) cfdict_ref = ',' 3 key/value pairs',
                    '(CFMutableDictionaryRef) mutable_dict_ref = ',' 12 key/value pairs',
                    '(CFArrayRef) cfarray_ref = ',' size=3',
                    '(CFMutableArrayRef) mutable_array_ref = ',' size=11'])

        self.expect('frame variable attrString mutableAttrString mutableGetConst',
                    substrs = ['(NSAttributedString *) attrString = ',' @"hello world from foo"',
                    '(NSAttributedString *) mutableAttrString = ',' @"hello world from foo"',
                    '(NSString *) mutableGetConst = ',' @"foo said this string needs to be very long so much longer than whatever other string has been seen ever before by anyone of the mankind that of course this is still not long enough given what foo our friend foo our lovely dearly friend foo desired of us so i am adding more stuff here for the sake of it and for the joy of our friend who is named guess what just foo. hence, dear friend foo, stay safe, your string is now  long enough to accommodate your testing need and I will make sure that if not we extend it with even more fuzzy random meaningless words pasted one after the other from a long tiresome friday evening spent working in my office. my office mate went home but I am still randomly typing just for the fun of seeing what happens of the length of a Mutable String in Cocoa if it goes beyond one byte.. so be it, dear foo"'])

        self.expect('frame variable immutableData mutableData data_ref mutable_data_ref mutable_string_ref',
                    substrs = ['(NSData *) immutableData = ',' 4 bytes',
                    '(NSData *) mutableData = ',' 14 bytes',
                    '(CFDataRef) data_ref = ',' 5 bytes',
                    '(CFMutableDataRef) mutable_data_ref = ',' 5 bytes',
                    '(CFMutableStringRef) mutable_string_ref = ',' @"Wish ya knew"'])

        self.expect('frame variable mutable_bag_ref cfbag_ref binheap_ref',
                    substrs = ['(CFMutableBagRef) mutable_bag_ref = ',' 17 items',
                    '(CFBagRef) cfbag_ref = ',' 15 items',
                    '(CFBinaryHeapRef) binheap_ref = ',' 21 items'])

        self.expect('frame variable cfurl_ref cfchildurl_ref cfgchildurl_ref',
                    substrs = ['(CFURLRef) cfurl_ref = ',' @"http://www.foo.bar/"',
                    'cfchildurl_ref = ',' @"page.html" (base path: @"http://www.foo.bar/")',
                    '(CFURLRef) cfgchildurl_ref = ',' @"?whatever" (base path: @"http://www.foo.bar/page.html")'])

        self.expect('frame variable nsurl nsurl2 nsurl3',
                    substrs = ['(NSURL *) nsurl = ',' @"http://www.foo.bar"',
                    '(NSURL *) nsurl2 =',' @"page.html" (base path: @"http://www.foo.bar")',
                    '(NSURL *) nsurl3 = ',' @"?whatever" (base path: @"http://www.foo.bar/page.html")'])

        self.expect('frame variable bundle_string bundle_url main_bundle',
                    substrs = ['(NSBundle *) bundle_string = ',' @"/System/Library/Frameworks/Accelerate.framework"',
                    '(NSBundle *) bundle_url = ',' @"/System/Library/Frameworks/Cocoa.framework"',
                    '(NSBundle *) main_bundle = ','test/functionalities/data-formatter/data-formatter-objc'])

        self.expect('frame variable except0 except1 except2 except3',
                    substrs = ['(NSException *) except0 = ',' @"TheGuyWhoHasNoName" @"cuz it\'s funny"',
                    '(NSException *) except1 = ',' @"TheGuyWhoHasNoName~1" @"cuz it\'s funny"',
                    '(NSException *) except2 = ',' @"TheGuyWhoHasNoName`2" @"cuz it\'s funny"',
                    '(NSException *) except3 = ',' @"TheGuyWhoHasNoName/3" @"cuz it\'s funny"'])

        self.expect('frame variable port',
                    substrs = ['(NSMachPort *) port = ',' mach port: '])

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
