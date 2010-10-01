"""
Test the lldb disassemble command.
"""

import os, time
import unittest2
import lldb
from lldbtest import *

@unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
class FoundationDisassembleTestCase(TestBase):

    mydir = "foundation"

    # rdar://problem/8504895
    # Crash while doing 'disassemble -n "-[NSNumber descriptionWithLocale:]"
    @unittest2.skipIf(TestBase.skipLongRunningTest(), "Skip this long running test")
    def test_foundation_disasm(self):
        """Do 'disassemble -n func' on each and every 'Code' symbol entry from the Foundation.framework."""
        self.buildDefault()
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)
        self.runCmd("run", RUN_SUCCEEDED)

        self.runCmd("image list")
        raw_output = self.res.GetOutput()
        # Grok the full path to the foundation framework.
        for line in raw_output.split(os.linesep):
            match = re.search(" (/.*/Foundation.framework/.*)$", line)
            if match:
                foundation_framework = match.group(1)
                break

        self.assertTrue(match, "Foundation.framework path located")
        self.runCmd("image dump symtab %s" % foundation_framework)
        raw_output = self.res.GetOutput()
        # Now, grab every 'Code' symbol and feed it into the 'disassemble -n func' command.
        for line in raw_output.split(os.linesep):
            # The symbol name is on the last column and trails the flag column which ends with '0000'.
            match = re.search(" Code         .+0000 (.+)$", line)
            if match:
                func = match.group(1)
                print "line:", line
                print "func:", func
                self.runCmd('disassemble -n "%s"' % func)
        

    def test_simple_disasm_with_dsym(self):
        """Test the lldb 'disassemble' command"""
        self.buildDsym()
        self.do_simple_disasm()

    def test_simple_disasm_with_dwarf(self):
        """Test the lldb 'disassemble' command"""
        self.buildDwarf()
        self.do_simple_disasm()

    def do_simple_disasm(self):
        """Do a bunch of simple disassemble commands."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Stop at +[NSString stringWithFormat:].
        self.expect("regexp-break +[NSString stringWithFormat:]", BREAKPOINT_CREATED,
            startstr = "Breakpoint created: 1: name = '+[NSString stringWithFormat:]', locations = 1")

        # Stop at -[MyString initWithNSString:].
        self.expect("breakpoint set -n '-[MyString initWithNSString:]'", BREAKPOINT_CREATED,
            startstr = "Breakpoint created: 2: name = '-[MyString initWithNSString:]', locations = 1")

        # Stop at the "description" selector.
        self.expect("breakpoint set -S description", BREAKPOINT_CREATED,
            startstr = "Breakpoint created: 3: name = 'description', locations = 1")

        # Stop at -[NSAutoreleasePool release].
        self.expect("regexp-break -[NSAutoreleasePool release]", BREAKPOINT_CREATED,
            startstr = "Breakpoint created: 4: name = '-[NSAutoreleasePool release]', locations = 1")

        self.runCmd("run", RUN_SUCCEEDED)

        # First stop is +[NSString stringWithFormat:].
        self.expect("thread backtrace", "Stop at +[NSString stringWithFormat:]",
            substrs = ["Foundation`+[NSString stringWithFormat:]"])

        # Do the disassemble for the currently stopped function.
        self.runCmd("disassemble")

        self.runCmd("process continue")

        # Followed by a.out`-[MyString initWithNSString:].
        self.expect("thread backtrace", "Stop at a.out`-[MyString initWithNSString:]",
            substrs = ["a.out`-[MyString initWithNSString:]"])

        # Do the disassemble for the currently stopped function.
        self.runCmd("disassemble")

        self.runCmd("process continue")

        # Followed by -[MyString description].
        self.expect("thread backtrace", "Stop at -[MyString description]",
            substrs = ["a.out`-[MyString description]"])

        # Do the disassemble for the currently stopped function.
        self.runCmd("disassemble")

        self.runCmd("process continue")

        # Followed by -[NSAutoreleasePool release].
        self.expect("thread backtrace", "Stop at -[NSAutoreleasePool release]",
            substrs = ["Foundation`-[NSAutoreleasePool release]"])

        # Do the disassemble for the currently stopped function.
        self.runCmd("disassemble")


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
