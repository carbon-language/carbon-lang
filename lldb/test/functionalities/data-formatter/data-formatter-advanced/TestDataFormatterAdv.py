"""
Test lldb data formatter subsystem.
"""

import os, time
import unittest2
import lldb
from lldbtest import *

class DataFormatterTestCase(TestBase):

    mydir = os.path.join("functionalities", "data-formatter", "data-formatter-advanced")

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

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        self.runCmd("type summary add -f \"pippo\" \"i_am_cool\"")

        self.runCmd("type summary add -f \"pluto\" -x \"i_am_cool[a-z]*\"")

        self.expect("frame variable cool_boy",
            substrs = ['pippo'])

        self.expect("frame variable cooler_boy",
            substrs = ['pluto'])

        self.runCmd("type summary delete i_am_cool")
        
        self.expect("frame variable cool_boy",
            substrs = ['pluto'])

        self.runCmd("type summary clear")
        
        self.runCmd("type summary add -f \"${var[]}\" -x \"int \\[[0-9]\\]")

        self.expect("frame variable int_array",
            substrs = ['1,2,3,4,5'])

        self.runCmd("type summary add -f \"${var[].integer}\" -x \"i_am_cool \\[[0-9]\\]")
        
        self.expect("frame variable cool_array",
            substrs = ['1,1,1,1,6'])

        self.runCmd("type summary clear")
            
        self.runCmd("type summary add -f \"${var[1-0]%x}\" \"int\"")
        
        self.expect("frame variable iAmInt",
            substrs = ['01'])
                
        self.runCmd("type summary add -f \"${var[0-1]%x}\" \"int\"")
        
        self.expect("frame variable iAmInt",
            substrs = ['01'])

        self.runCmd("type summary clear")

        self.runCmd("type summary add -f \"${var[0-1]%x}\" int")
        self.runCmd("type summary add -f \"${var[0-31]%x}\" float")
                    
        self.expect("frame variable *pointer",
            substrs = ['0x',
                       '2'])

        self.expect("frame variable cool_array[3].floating",
            substrs = ['0x'])
                    
        self.runCmd("type summary add -f \"low bits are ${*var[0-1]} tgt is ${*var}\" \"int *\"")

        self.expect("frame variable pointer",
            substrs = ['low bits are',
                       'tgt is 6'])

        self.runCmd("type summary add -f \"${*var[0-1]}\" -x \"int \[[0-9]\]\"")

        self.expect("frame variable int_array",
            substrs = ['3'])

        self.runCmd("type summary clear")
            
        self.runCmd("type summary add -f \"${var[0-1]}\" -x \"int \[[0-9]\]\"")

        self.expect("frame variable int_array",
            substrs = ['1,2'])

        self.runCmd("type summary clear")

        self.runCmd("type summary add -c -x \"i_am_cool \[[0-9]\]\"")
        self.runCmd("type summary add -c i_am_cool")

        self.expect("frame variable cool_array",
            substrs = ['[0]',
                       '[1]',
                       '[2]',
                       '[3]',
                       '[4]',
                       'integer',
                       'character',
                       'floating'])

        self.runCmd("type summary add -f \"int = ${*var.int_pointer}, float = ${*var.float_pointer}\" IWrapPointers")

        self.expect("frame variable wrapper",
            substrs = ['int = 4',
                       'float = 1.1'])

        self.runCmd("type summary add -f \"low bits = ${*var.int_pointer[2]}\" IWrapPointers -p")
        
        self.expect("frame variable wrapper",
            substrs = ['low bits = 1'])
        
        self.expect("frame variable *wrap_pointer",
            substrs = ['low bits = 1'])

        self.runCmd("type summary clear")

        self.runCmd("type summary add -f \"${var[0][0-2]%hex}\" -x \"int \[[0-9]\]\"")

        self.expect("frame variable int_array",
            substrs = ['0x',
                       '7'])

        self.runCmd("type summary clear")

        self.runCmd("type summary add -f \"${*var[].x[0-3]%hex} is a bitfield on a set of integers\" -x \"SimpleWithPointers \[[0-9]\]\"")
        self.runCmd("type summary add -f \"${*var.sp.x[0-2]} are low bits of integer ${*var.sp.x}. If I pretend it is an array I get ${var.sp.x[0-5]}\" Couple")

        self.expect("frame variable couple",
            substrs = ['1 are low bits of integer 9.',
                       'If I pretend it is an array I get [9,'])

        self.expect("frame variable sparray",
            substrs = ['[0x0000000f,0x0000000c,0x00000009]'])
        


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
