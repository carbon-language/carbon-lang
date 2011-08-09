"""
Test lldb data formatter subsystem.
"""

import os, time
import unittest2
import lldb
from lldbtest import *

class DataFormatterTestCase(TestBase):

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

        # check that we do not get the extra vars and that we cache results
        self.expect("frame variable f00_1", matching=False,
                    substrs = ['looking for',
                               'b = 1']);

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
        
        # delete the synth and check that we get good output
        self.runCmd("type synth delete foo")
        self.expect("frame variable f00_1",
                    substrs = ['a = 280',
                               'b = 1',
                               'r = 33']);

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

        # now start playing with STL containers
        # having std::<class_type> here is a workaround for rdar://problem/9835692
        
        
        # std::vector
        self.runCmd("script from StdVectorSynthProvider import *")
        self.runCmd("type synth add -l StdVectorSynthProvider std::int_vect int_vect")
        self.runCmd("type synth add -l StdVectorSynthProvider std::string_vect string_vect")

        self.runCmd("n")

        # empty vectors (and storage pointers SHOULD BOTH BE NULL..)
        self.expect("frame variable numbers",
            substrs = ['numbers = {}'])

        self.runCmd("n")
        
        # first value added
        self.expect("frame variable numbers",
                    substrs = ['numbers = {',
                               '[0] = 1',
                               '}'])

        # add some more data
        self.runCmd("n");self.runCmd("n");self.runCmd("n");
    
        self.expect("frame variable numbers",
                    substrs = ['numbers = {',
                               '[0] = 1',
                               '[1] = 12',
                               '[2] = 123',
                               '[3] = 1234',
                               '}'])

        # add some more data
        self.runCmd("n");self.runCmd("n");self.runCmd("n");

        self.expect("frame variable numbers",
                    substrs = ['numbers = {',
                               '[0] = 1',
                               '[1] = 12',
                               '[2] = 123',
                               '[3] = 1234',
                               '[4] = 12345',
                               '[5] = 123456',
                               '[6] = 1234567',
                               '}'])
        
        # check access-by-index
        self.expect("frame variable numbers[0]",
                    substrs = ['1']);
        self.expect("frame variable numbers[1]",
                    substrs = ['12']);
        self.expect("frame variable numbers[2]",
                    substrs = ['123']);
        self.expect("frame variable numbers[3]",
                    substrs = ['1234']);
        
        # but check that expression does not rely on us
        # (when expression gets to call into STL code correctly, we will have to find
        # another way to check this)
        self.expect("expression numbers[6]", matching=False, error=True,
            substrs = ['1234567'])

        # clear out the vector and see that we do the right thing once again
        self.runCmd("n")

        self.expect("frame variable numbers",
            substrs = ['numbers = {}'])

        self.runCmd("n")

        # first value added
        self.expect("frame variable numbers",
                    substrs = ['numbers = {',
                               '[0] = 7',
                               '}'])

        # check if we can display strings
        self.runCmd("n")
        self.runCmd("n")
        self.runCmd("n")
        self.runCmd("n")

        self.expect("frame variable strings",
            substrs = ['goofy',
                       'is',
                       'smart'])

        # test summaries based on synthetic children
        self.runCmd("type summary add std::string_vect string_vect -f \"vector has ${svar%#} items\" -e")
        self.expect("frame variable strings",
                    substrs = ['vector has 3 items',
                               'goofy',
                               'is',
                               'smart'])

        self.runCmd("n");

        self.expect("frame variable strings",
                    substrs = ['vector has 4 items'])
        
        # check access-by-index
        self.expect("frame variable strings[0]",
                    substrs = ['goofy']);
        self.expect("frame variable strings[1]",
                    substrs = ['is']);
        
        # but check that expression does not rely on us
        # (when expression gets to call into STL code correctly, we will have to find
        # another way to check this)
        self.expect("expression strings[0]", matching=False, error=True,
                    substrs = ['goofy'])

        self.runCmd("n")

        self.expect("frame variable strings",
            substrs = ['vector has 0 items'])

        # now test std::list
        self.runCmd("script from StdListSynthProvider import *")

        self.runCmd("n")

        self.runCmd("frame variable numbers_list -T")
        self.runCmd("type synth add std::int_list std::string_list int_list string_list -l StdListSynthProvider")
        self.runCmd("type summary add std::int_list std::string_list int_list string_list -f \"list has ${svar%#} items\" -e")
        self.runCmd("type format add -f hex int")

        self.expect("frame variable numbers_list",
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

        # let's prettify string display
        self.runCmd("type summary add -f \"${var._M_dataplus._M_p}\" std::string std::basic_string<char> \"std::basic_string<char,std::char_traits<char>,std::allocator<char> >\"")

        self.expect("frame variable text_list",
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
        
        self.runCmd("script from StdMapSynthProvider import *")
        self.runCmd("type summary add -x \"std::map<\" -f \"map has ${svar%#} items\" -e")
        self.runCmd("type synth add -x \"std::map<\" -l StdMapSynthProvider")


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

        self.expect('frame variable ii',
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

        #self.runCmd("type summary add std::strint_map strint_map -f \"map has ${svar%#} items\" -e")
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

        self.expect('frame variable si',
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
        
        #self.runCmd("type summary add std::intstr_map intstr_map -f \"map has ${svar%#} items\" -e")
        #self.runCmd("type synth add std::intstr_map intstr_map -l StdMapSynthProvider")

        self.expect('frame variable is',
                    substrs = ['map has 0 items',
                               '{}'])

        self.runCmd("n");self.runCmd("n");self.runCmd("n");self.runCmd("n");

        self.expect('frame variable is',
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
        
        #self.runCmd("type summary add std::strstr_map strstr_map -f \"map has ${svar%#} items\" -e")
        #self.runCmd("type synth add std::strstr_map strstr_map -l StdMapSynthProvider")

        self.expect('frame variable ss',
                    substrs = ['map has 0 items',
                               '{}'])

        self.runCmd("n");self.runCmd("n");self.runCmd("n");self.runCmd("n");

        self.expect('frame variable ss',
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
