"""
Test some SBModule and SBSection APIs.
"""

import os, time
import re
import unittest2
import lldb
from lldbtest import *
from lldbutil import symbol_iter, symbol_type_to_str

class ModuleAndSectionAPIsTestCase(TestBase):

    mydir = os.path.join("python_api", "module_section")

    @python_api_test
    def test_module_and_section(self):
        """Test module and section APIs."""
        self.buildDefault()
        self.module_and_section()

    def module_and_section(self):
        exe = os.path.join(os.getcwd(), "a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)
        self.assertTrue(target.GetNumModules() > 0)

        # Hide stdout if not running with '-t' option.
        if not self.TraceOn():
            self.HideStdout()

        print "Number of modules for the target: %d" % target.GetNumModules()
        for module in target.module_iter():
            print module

        # Get the executable module at index 0.
        exe_module = target.GetModuleAtIndex(0)

        print "Exe module: %s" % repr(exe_module)
        print "Number of sections: %d" % exe_module.GetNumSections()
        INDENT = ' ' * 4
        INDENT2 = INDENT * 2
        for sec in exe_module.section_iter():
            print sec
            print INDENT + "Number of subsections: %d" % sec.GetNumSubSections()
            if sec.GetNumSubSections() == 0:
                for sym in exe_module.symbol_in_section_iter(sec):
                    print INDENT + repr(sym)
                    print INDENT + "symbol type: %s" % symbol_type_to_str(sym.GetType())
            else:
                for subsec in sec:
                    print INDENT + repr(subsec)
                    # Now print the symbols belonging to the subsection....
                    for sym in exe_module.symbol_in_section_iter(subsec):
                        print INDENT2 + repr(sym)
                        print INDENT2 + "symbol type: %s" % symbol_type_to_str(sym.GetType())


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
