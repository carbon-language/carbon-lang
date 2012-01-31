"""
Test lldb data formatter subsystem.
"""

import os, time
import unittest2
import lldb
from lldbtest import *

class StdListDataFormatterTestCase(TestBase):

    mydir = os.path.join("functionalities", "data-formatter", "data-formatter-stl", "list")

    #rdar://problem/10334911
    @unittest2.expectedFailure
    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_with_dsym_and_run_command(self):
        """Test data formatter commands."""
        self.buildDsym()
        self.data_formatter_commands()

    #rdar://problem/10334911
    @unittest2.expectedFailure
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
            startstr = "Breakpoint created: 1: file ='main.cpp', line = %d" %
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
            self.runCmd("settings set target.max-children-count 256", check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        self.runCmd("frame variable numbers_list -T")
        #self.runCmd("type synth add std::int_list std::string_list int_list string_list -l StdListSynthProvider")
        self.runCmd("type summary add std::int_list std::string_list int_list string_list --summary-string \"list has ${svar%#} items\" -e")
        self.runCmd("type format add -f hex int")

        self.expect("frame variable numbers_list --raw", matching=False,
                    substrs = ['list has 0 items',
                               '{}'])

        self.expect("frame variable numbers_list",
                    substrs = ['list has 0 items',
                               '{}'])

        self.expect("p numbers_list",
                    substrs = ['list has 0 items',
                               '{}'])

        self.runCmd("n")

        self.expect("frame variable numbers_list",
                    substrs = ['list has 1 items',
                               '[0] = ',
                               '0x12345678'])

        self.runCmd("n");self.runCmd("n");self.runCmd("n");

        self.expect("frame variable numbers_list",
                    substrs = ['list has 4 items',
                               '[0] = ',
                               '0x12345678',
                               '[1] =',
                               '0x11223344',
                               '[2] =',
                               '0xbeeffeed',
                               '[3] =',
                               '0x00abba00'])

        self.runCmd("n");self.runCmd("n");

        self.expect("frame variable numbers_list",
                    substrs = ['list has 6 items',
                               '[0] = ',
                               '0x12345678',
                               '0x11223344',
                               '0xbeeffeed',
                               '0x00abba00',
                               '[4] =',
                               '0x0abcdef0',
                               '[5] =',
                               '0x0cab0cab'])

        self.expect("p numbers_list",
                    substrs = ['list has 6 items',
                               '[0] = ',
                               '0x12345678',
                               '0x11223344',
                               '0xbeeffeed',
                               '0x00abba00',
                               '[4] =',
                               '0x0abcdef0',
                               '[5] =',
                               '0x0cab0cab'])

        # check access-by-index
        self.expect("frame variable numbers_list[0]",
                    substrs = ['0x12345678']);
        self.expect("frame variable numbers_list[1]",
                    substrs = ['0x11223344']);
        
        # but check that expression does not rely on us
        self.expect("expression numbers_list[0]", matching=False, error=True,
                    substrs = ['0x12345678'])

        self.runCmd("n")
            
        self.expect("frame variable numbers_list",
                    substrs = ['list has 0 items',
                               '{}'])

        self.runCmd("n");self.runCmd("n");self.runCmd("n");self.runCmd("n");
            
        self.expect("frame variable numbers_list",
                    substrs = ['list has 4 items',
                               '[0] = ', '1',
                               '[1] = ', '2',
                               '[2] = ', '3',
                               '[3] = ', '4'])            

        self.runCmd("type format delete int")

        self.runCmd("n")
            
        self.expect("frame variable text_list",
            substrs = ['list has 0 items',
                       '{}'])
        
        self.runCmd("n");self.runCmd("n");self.runCmd("n");self.runCmd("n");

        self.expect("frame variable text_list",
                    substrs = ['list has 4 items',
                               '[0]', 'goofy',
                               '[1]', 'is',
                               '[2]', 'smart',
                               '[3]', '!!!'])

        self.expect("p text_list",
                    substrs = ['list has 4 items',
                               '[0] = \"goofy\"',
                               '[1] = \"is\"',
                               '[2] = \"smart\"',
                               '[3] = \"!!!\"'])
        
        # check access-by-index
        self.expect("frame variable text_list[0]",
                    substrs = ['goofy']);
        self.expect("frame variable text_list[3]",
                    substrs = ['!!!']);
        
        # but check that expression does not rely on us
        self.expect("expression text_list[0]", matching=False, error=True,
                    substrs = ['goofy'])

        # now std::map<K,V>
        # also take a chance to test regex synth here

        self.runCmd("n")
        self.runCmd("frame variable ii -T")
        
        #self.runCmd("script from StdMapSynthProvider import *")
        self.runCmd("type summary add -x \"std::map<\" --summary-string \"map has ${svar%#} items\" -e") 
        
        #import time
        #time.sleep(30)
        
        #self.runCmd("type synth add -x \"std::map<\" -l StdMapSynthProvider")


        self.expect('frame variable ii',
            substrs = ['map has 0 items',
                       '{}'])

        self.runCmd("n");self.runCmd("n");

        self.expect('frame variable ii',
                    substrs = ['map has 2 items',
                               '[0] = {',
                               'first = 0',
                               'second = 0',
                               '[1] = {',
                               'first = 1',
                               'second = 1'])

        self.runCmd("n");self.runCmd("n");

        self.expect('frame variable ii',
                    substrs = ['map has 4 items',
                               '[2] = {',
                               'first = 2',
                               'second = 0',
                               '[3] = {',
                               'first = 3',
                               'second = 1'])

        self.runCmd("n");self.runCmd("n");
        self.runCmd("n");self.runCmd("n");self.runCmd("n");

        self.expect("frame variable ii",
                    substrs = ['map has 9 items',
                               '[5] = {',
                               'first = 5',
                               'second = 0',
                               '[7] = {',
                               'first = 7',
                               'second = 1'])
        
        self.expect("p ii",
                    substrs = ['map has 9 items',
                               '[5] = {',
                               'first = 5',
                               'second = 0',
                               '[7] = {',
                               'first = 7',
                               'second = 1'])

        # check access-by-index
        self.expect("frame variable ii[0]",
                    substrs = ['first = 0',
                               'second = 0']);
        self.expect("frame variable ii[3]",
                    substrs = ['first =',
                               'second =']);
        
        # but check that expression does not rely on us
        self.expect("expression ii[0]", matching=False, error=True,
                    substrs = ['first = 0'])

        self.runCmd("n")
        
        self.expect('frame variable ii',
                    substrs = ['map has 0 items',
                               '{}'])
        
        self.runCmd("n")
        self.runCmd("frame variable si -T")

        #self.runCmd("type summary add std::strint_map strint_map --summary-string \"map has ${svar%#} items\" -e")
        #self.runCmd("type synth add std::strint_map strint_map -l StdMapSynthProvider")
        
        self.expect('frame variable si',
                    substrs = ['map has 0 items',
                               '{}'])

        self.runCmd("n")

        self.expect('frame variable si',
                    substrs = ['map has 1 items',
                               '[0] = ',
                               'first = \"zero\"',
                               'second = 0'])

        self.runCmd("n");self.runCmd("n");self.runCmd("n");self.runCmd("n");

        self.expect("frame variable si",
                    substrs = ['map has 5 items',
                               '[0] = ',
                               'first = \"zero\"',
                               'second = 0',
                                '[1] = ',
                                'first = \"one\"',
                                'second = 1',
                                '[2] = ',
                                'first = \"two\"',
                                'second = 2',
                                '[3] = ',
                                'first = \"three\"',
                                'second = 3',
                                '[4] = ',
                                'first = \"four\"',
                                'second = 4'])

        self.expect("p si",
                    substrs = ['map has 5 items',
                               '[0] = ',
                               'first = \"zero\"',
                               'second = 0',
                               '[1] = ',
                               'first = \"one\"',
                               'second = 1',
                               '[2] = ',
                               'first = \"two\"',
                               'second = 2',
                               '[3] = ',
                               'first = \"three\"',
                               'second = 3',
                               '[4] = ',
                               'first = \"four\"',
                               'second = 4'])

        # check access-by-index
        self.expect("frame variable si[0]",
                    substrs = ['first = ', 'four',
                               'second = 4']);
        
        # but check that expression does not rely on us
        self.expect("expression si[0]", matching=False, error=True,
                    substrs = ['first = ', 'zero'])

        self.runCmd("n")
        
        self.expect('frame variable si',
                    substrs = ['map has 0 items',
                               '{}'])

        self.runCmd("n")
        self.runCmd("frame variable is -T")
        
        #self.runCmd("type summary add std::intstr_map intstr_map --summary-string \"map has ${svar%#} items\" -e")
        #self.runCmd("type synth add std::intstr_map intstr_map -l StdMapSynthProvider")

        self.expect('frame variable is',
                    substrs = ['map has 0 items',
                               '{}'])

        self.runCmd("n");self.runCmd("n");self.runCmd("n");self.runCmd("n");

        self.expect("frame variable is",
                    substrs = ['map has 4 items',
                               '[0] = ',
                               'second = \"goofy\"',
                               'first = 0',
                               '[1] = ',
                               'second = \"is\"',
                               'first = 1',
                               '[2] = ',
                               'second = \"smart\"',
                               'first = 2',
                               '[3] = ',
                               'second = \"!!!\"',
                               'first = 3'])
        
        self.expect("p is",
                    substrs = ['map has 4 items',
                               '[0] = ',
                               'second = \"goofy\"',
                               'first = 0',
                               '[1] = ',
                               'second = \"is\"',
                               'first = 1',
                               '[2] = ',
                               'second = \"smart\"',
                               'first = 2',
                               '[3] = ',
                               'second = \"!!!\"',
                               'first = 3'])

        # check access-by-index
        self.expect("frame variable is[0]",
                    substrs = ['first = ', '0',
                               'second =', 'goofy']);
        
        # but check that expression does not rely on us
        self.expect("expression is[0]", matching=False, error=True,
                    substrs = ['first = ', 'goofy'])

        self.runCmd("n")
        
        self.expect('frame variable is',
                    substrs = ['map has 0 items',
                               '{}'])

        self.runCmd("n")
        self.runCmd("frame variable ss -T")
        
        #self.runCmd("type summary add std::strstr_map strstr_map --summary-string \"map has ${svar%#} items\" -e")
        #self.runCmd("type synth add std::strstr_map strstr_map -l StdMapSynthProvider")

        self.expect('frame variable ss',
                    substrs = ['map has 0 items',
                               '{}'])

        self.runCmd("n");self.runCmd("n");self.runCmd("n");self.runCmd("n");

        self.expect("frame variable ss",
                    substrs = ['map has 4 items',
                               '[0] = ',
                               'second = \"hello\"',
                               'first = \"ciao\"',
                               '[1] = ',
                               'second = \"house\"',
                               'first = \"casa\"',
                               '[2] = ',
                               'second = \"cat\"',
                               'first = \"gatto\"',
                               '[3] = ',
                               'second = \"..is always a Mac!\"',
                               'first = \"a Mac..\"'])
        
        self.expect("p ss",
                    substrs = ['map has 4 items',
                               '[0] = ',
                               'second = \"hello\"',
                               'first = \"ciao\"',
                               '[1] = ',
                               'second = \"house\"',
                               'first = \"casa\"',
                               '[2] = ',
                               'second = \"cat\"',
                               'first = \"gatto\"',
                               '[3] = ',
                               'second = \"..is always a Mac!\"',
                               'first = \"a Mac..\"'])

        # check access-by-index
        self.expect("frame variable ss[3]",
                    substrs = ['gatto', 'cat']);
        
        # but check that expression does not rely on us
        self.expect("expression ss[3]", matching=False, error=True,
                    substrs = ['gatto'])

        self.runCmd("n")
        
        self.expect('frame variable ss',
                    substrs = ['map has 0 items',
                               '{}'])

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
