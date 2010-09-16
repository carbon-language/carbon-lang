"""Set breakpoint on objective-c class and instance methods in foundation."""

import os, time
import unittest2
import lldb
from lldbtest import *

@unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
class FoundationTestCase(TestBase):

    mydir = "foundation"

    def test_with_dsym(self):
        """Test setting objc breakpoints using 'regexp-break' and 'breakpoint set'."""
        self.buildDsym()
        self.break_on_objc_methods()

    def test_with_dwarf(self):
        """Test setting objc breakpoints using 'regexp-break' and 'breakpoint set'."""
        self.buildDwarf()
        self.break_on_objc_methods()

    def break_on_objc_methods(self):
        """Test setting objc breakpoints using 'regexp-break' and 'breakpoint set'."""
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

        self.runCmd("process continue")

        # Followed by a.out`-[MyString initWithNSString:].
        self.expect("thread backtrace", "Stop at a.out`-[MyString initWithNSString:]",
            substrs = ["a.out`-[MyString initWithNSString:]"])

        self.runCmd("process continue")

        # Followed by -[MyString description].
        self.expect("thread backtrace", "Stop at -[MyString description]",
            substrs = ["a.out`-[MyString description]"])

        self.runCmd("process continue")

        # Followed by -[NSAutoreleasePool release].
        self.expect("thread backtrace", "Stop at -[NSAutoreleasePool release]",
            substrs = ["Foundation`-[NSAutoreleasePool release]"])


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
