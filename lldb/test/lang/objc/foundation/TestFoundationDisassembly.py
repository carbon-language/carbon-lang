"""
Test the lldb disassemble command on foundation framework.
"""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

@unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
class FoundationDisassembleTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    # rdar://problem/8504895
    # Crash while doing 'disassemble -n "-[NSNumber descriptionWithLocale:]"
    @unittest2.skipIf(TestBase.skipLongRunningTest(), "Skip this long running test")
    def test_foundation_disasm(self):
        """Do 'disassemble -n func' on each and every 'Code' symbol entry from the Foundation.framework."""
        self.buildDefault()
        
        # Enable synchronous mode
        self.dbg.SetAsync(False)
        
        # Create a target by the debugger.
        target = self.dbg.CreateTarget("a.out")
        self.assertTrue(target, VALID_TARGET)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple (None, environment, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

        for module in target.modules:
            if module.file.basename == "Foundation":
                foundation_framework = module.file.fullpath
                break

        self.assertTrue(match, "Foundation.framework path located")
        self.runCmd("image dump symtab %s" % foundation_framework)
        raw_output = self.res.GetOutput()
        # Now, grab every 'Code' symbol and feed it into the command:
        # 'disassemble -n func'.
        #
        # The symbol name is on the last column and trails the flag column which
        # looks like '0xhhhhhhhh', i.e., 8 hexadecimal digits.
        codeRE = re.compile(r"""
                             \ Code\ {9}    # ' Code' followed by 9 SPCs,
                             .*             # the wildcard chars,
                             0x[0-9a-f]{8}  # the flag column, and
                             \ (.+)$        # finally the function symbol.
                             """, re.VERBOSE)
        for line in raw_output.split(os.linesep):
            match = codeRE.search(line)
            if match:
                func = match.group(1)
                #print "line:", line
                #print "func:", func
                self.runCmd('disassemble -n "%s"' % func)
        

    @dsym_test
    def test_simple_disasm_with_dsym(self):
        """Test the lldb 'disassemble' command"""
        self.buildDsym()
        self.do_simple_disasm()

    @dwarf_test
    def test_simple_disasm_with_dwarf(self):
        """Test the lldb 'disassemble' command"""
        self.buildDwarf()
        self.do_simple_disasm()

    def do_simple_disasm(self):
        """Do a bunch of simple disassemble commands."""
        
        # Create a target by the debugger.
        target = self.dbg.CreateTarget("a.out")
        self.assertTrue(target, VALID_TARGET)

        print target
        for module in target.modules:
            print module

        # Stop at +[NSString stringWithFormat:].
        symbol_name = "+[NSString stringWithFormat:]"
        break_results = lldbutil.run_break_set_command (self, "_regexp-break %s"%(symbol_name))
        
        lldbutil.check_breakpoint_result (self, break_results, symbol_name=symbol_name, num_locations=1)

        # Stop at -[MyString initWithNSString:].
        lldbutil.run_break_set_by_symbol (self, '-[MyString initWithNSString:]', num_expected_locations=1, sym_exact=True)

        # Stop at the "description" selector.
        lldbutil.run_break_set_by_selector (self, 'description', num_expected_locations=1, module_name='a.out')

        # Stop at -[NSAutoreleasePool release].
        break_results = lldbutil.run_break_set_command (self, "_regexp-break -[NSAutoreleasePool release]")
        lldbutil.check_breakpoint_result (self, break_results, symbol_name='-[NSAutoreleasePool release]', num_locations=1)

        self.runCmd("run", RUN_SUCCEEDED)

        # First stop is +[NSString stringWithFormat:].
        self.expect("thread backtrace", "Stop at +[NSString stringWithFormat:]",
            substrs = ["Foundation`+[NSString stringWithFormat:]"])

        # Do the disassemble for the currently stopped function.
        self.runCmd("disassemble -f")

        self.runCmd("process continue")
        # Skip another breakpoint for +[NSString stringWithFormat:].
        self.runCmd("process continue")

        # Followed by a.out`-[MyString initWithNSString:].
        self.expect("thread backtrace", "Stop at a.out`-[MyString initWithNSString:]",
            substrs = ["a.out`-[MyString initWithNSString:]"])

        # Do the disassemble for the currently stopped function.
        self.runCmd("disassemble -f")

        self.runCmd("process continue")

        # Followed by -[MyString description].
        self.expect("thread backtrace", "Stop at -[MyString description]",
            substrs = ["a.out`-[MyString description]"])

        # Do the disassemble for the currently stopped function.
        self.runCmd("disassemble -f")

        self.runCmd("process continue")
        # Skip another breakpoint for -[MyString description].
        self.runCmd("process continue")

        # Followed by -[NSAutoreleasePool release].
        self.expect("thread backtrace", "Stop at -[NSAutoreleasePool release]",
            substrs = ["Foundation`-[NSAutoreleasePool release]"])

        # Do the disassemble for the currently stopped function.
        self.runCmd("disassemble -f")


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
