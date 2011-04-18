"""
Test that template instaniations of std::vector<long> and <short> in the same module have the correct types.
"""

import unittest2
import lldb
import lldbutil
from lldbtest import *

class UniqueTypesTestCase(TestBase):

    mydir = "unique-types"

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_with_dsym(self):
        """Test for unique types of std::vector<long> and std::vector<short>."""
        self.buildDsym()
        self.unique_types()

    def test_with_dwarf(self):
        """Test for unique types of std::vector<long> and std::vector<short>."""
        self.buildDwarf()
        self.unique_types()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number inside main.cpp.
        self.line = line_number("main.cpp",
          "// Set breakpoint here to verify that std::vector 'longs' and 'shorts' have unique types.")

    def unique_types(self):
        """Test for unique types of std::vector<long> and std::vector<short>."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        self.expect("breakpoint set -f main.cpp -l %d" % self.line,
                    BREAKPOINT_CREATED,
            startstr = "Breakpoint created: 1: file ='main.cpp', line = %d, locations = 1" %
                        self.line)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        if self.getCompiler().endswith('clang'):
            import re
            clang_version_output = system([lldbutil.which(self.getCompiler()), "-v"])[1]
            #print "my output:", clang_version_output
            for line in clang_version_output.split(os.linesep):
                m = re.search('clang version ([0-9]+)\.', line)
                #print "line:", line
                if m:
                    clang_version = int(m.group(1))
                    #print "clang version:", clang_version
                    if clang_version < 3:
                        self.skipTest("rdar://problem/9173060 lldb hangs while running unique-types for clang version < 3")

        # Do a "frame variable -t longs" and verify "long" is in each line of output.
        self.runCmd("frame variable -t longs")
        output = self.res.GetOutput()
        for x in [line.strip() for line in output.split(os.linesep)]:
            # Skip empty line or closing brace.
            if not x or x == '}':
                continue
            self.expect(x, "Expect type 'long'", exe=False,
                substrs = ['long'])

        # Do a "frame variable -t shorts" and verify "short" is in each line of output.
        self.runCmd("frame variable -t shorts")
        output = self.res.GetOutput()
        for x in [line.strip() for line in output.split(os.linesep)]:
            # Skip empty line or closing brace.
            if not x or x == '}':
                continue
            self.expect(x, "Expect type 'short'", exe=False,
                substrs = ['short'])
        

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
