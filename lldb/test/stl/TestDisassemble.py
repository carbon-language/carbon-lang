"""
Test the lldb disassemble command.
"""

import os, time
import unittest2
import lldb
from lldbtest import *

@unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
class StdCXXDisassembleTestCase(TestBase):

    mydir = "stl"

    @unittest2.skipIf(TestBase.skipLongRunningTest(), "Skip this long running test")
    def test_stdcxx_disasm(self):
        """Do 'disassemble' on each and every 'Code' symbol entry from the std c++ lib."""
        self.buildDefault()
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break on line 13 of main.cpp.
        self.expect("breakpoint set -f main.cpp -l 13", BREAKPOINT_CREATED,
            startstr = "Breakpoint created: 1: file ='main.cpp', line = 13, locations = 1")

        self.runCmd("run", RUN_SUCCEEDED)

        # Now let's get the target as well as the process objects.
        target = self.dbg.GetSelectedTarget()
        process = target.GetProcess()

        # The process should be in a 'Stopped' state.
        self.expect(repr(process), STOPPED_DUE_TO_BREAKPOINT, exe=False,
            substrs = ["a.out",
                       "state: Stopped"])

        # Iterate through the available modules, looking for stdc++ library...
        for i in range(target.GetNumModules()):
            module = target.GetModuleAtIndex(i)
            fs = module.GetFileSpec()
            if (fs.GetFilename().startswith("libstdc++")):
                lib_stdcxx = repr(fs)
                break

        # At this point, lib_stdcxx is the full path to the stdc++ library and
        # module is the corresponding SBModule.

        self.expect(fs.GetFilename(), "Libraray StdC++ is located", exe=False,
            substrs = ["libstdc++"])


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
