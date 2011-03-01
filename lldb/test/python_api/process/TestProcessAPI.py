"""
Test symbol table access for main.m.
"""

import os, time
import unittest2
import lldb
from lldbtest import *

@unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
class ProcessAPITestCase(TestBase):

    mydir = os.path.join("python_api", "process")

    symbols_list = ['main',
                    'my_char'
                    ]

    @python_api_test
    def test_with_dsym_and_python_api(self):
        """Test Python process APIs."""
        self.buildDsym()
        self.process_api()

    @python_api_test
    def test_with_dwarf_and_python_api(self):
        """Test Python process APIs."""
        self.buildDwarf()
        self.process_api()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number("main.cpp", "// Set break point at this line and check variable 'my_char'.")

    def process_api(self):
        """Test Python process APIs."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target.IsValid(), VALID_TARGET)

        breakpoint = target.BreakpointCreateByLocation("main.cpp", self.line)
        self.assertTrue(breakpoint.IsValid(), VALID_BREAKPOINT)

        # Launch the process, and do not stop at the entry point.
        error = lldb.SBError()
        self.process = target.Launch (self.dbg.GetListener(), None, None, os.ctermid(), os.ctermid(), os.ctermid(), None, 0, False, error)

        thread = self.process.GetThreadAtIndex(0);
        frame = thread.GetFrameAtIndex(0);

        # Get the SBValue for the global variable 'my_char'.
        val = frame.FindValue("my_char", lldb.eValueTypeVariableGlobal)
        location = int(val.GetLocation(frame), 16)
        self.DebugSBValue(frame, val)

        # Due to the typemap magic (see lldb.swig), we pass in 1 to ReadMemory and
        # expect to get a Python string as the result object!
        content = self.process.ReadMemory(location, 1, error)
        print "content:", content

        self.expect(content, "Result from SBProcess.ReadMemory() matches our expected output: 'x'",
                    exe=False,
            startstr = 'x')

        #
        # Exercise Python APIs to access the symbol table entries.
        #

        # Create the filespec by which to locate our a.out module.
        filespec = lldb.SBFileSpec(exe, False)

        module = target.FindModule(filespec)
        self.assertTrue(module.IsValid(), VALID_MODULE)

        # Create the set of known symbols.  As we iterate through the symbol
        # table, remove the symbol from the set if it is a known symbol.
        expected_symbols = set(self.symbols_list)
        from lldbutil import lldb_iter
        for symbol in lldb_iter(module, 'GetNumSymbols', 'GetSymbolAtIndex'):
            self.assertTrue(symbol.IsValid(), VALID_SYMBOL)
            #print "symbol:", symbol
            name = symbol.GetName()
            if name in expected_symbols:
                #print "Removing %s from known_symbols %s" % (name, expected_symbols)
                expected_symbols.remove(name)

        # At this point, the known_symbols set should have become an empty set.
        # If not, raise an error.
        #print "symbols unaccounted for:", expected_symbols
        self.assertTrue(len(expected_symbols) == 0,
                        "All the known symbols are accounted for")


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
