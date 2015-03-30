"""
Test some lldb command abbreviations.
"""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

class DisassemblyTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipUnlessDarwin
    @dsym_test
    def test_with_dsym (self):
        self.buildDsym ()
        self.disassemble_breakpoint ()

    @dwarf_test
    @expectedFailureLinux # llgs Handle_m returns target memory with breakpoints
    def test_with_dwarf (self):
        self.buildDwarf ()
        self.disassemble_breakpoint ()

    def disassemble_breakpoint (self):
        exe = os.path.join (os.getcwd(), "a.out")
        self.expect("file " + exe,
                    patterns = [ "Current executable set to .*a.out.*" ])

        match_object = lldbutil.run_break_set_command (self, "br s -n sum")
        lldbutil.check_breakpoint_result (self, match_object, symbol_name='sum', symbol_match_exact=False, num_locations=1)

        self.expect("run",
                    patterns = [ "Process .* launched: "])

        self.runCmd("dis -f")
        disassembly = self.res.GetOutput()

        # ARCH, if not specified, defaults to x86_64.
        if self.getArchitecture() in ["", 'x86_64', 'i386']:
            # make sure that the software breakpoint has been removed
            self.assertFalse("int3" in disassembly)
            # make sure a few reasonable assembly instructions are here
            self.expect(disassembly, exe=False,
                        startstr = "a.out`sum(int, int)",
                        substrs = [' mov',
                                   ' addl ',
                                   'ret'])
        else:
            # TODO please add your arch here
            self.fail('unimplemented for arch = "{arch}"'.format(arch=self.getArchitecture()))

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()

