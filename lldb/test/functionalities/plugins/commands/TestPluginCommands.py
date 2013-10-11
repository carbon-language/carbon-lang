"""
Test that plugins that load commands work correctly.
"""

import os, time
import re
import unittest2
import lldb
from lldbtest import *
import lldbutil

class PluginCommandTestCase(TestBase):

    mydir = os.path.join("functionalities", "plugins", "commands")

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        self.lib_dir = os.environ["LLDB_LIB_DIR"]

    @expectedFailureFreeBSD('llvm.org/pr17430')
    @skipIfi386 # This test links against liblldb.so. Thus, the test requires a 32-bit liblldb.so.
    def test_load_plugin(self):
        """Test that plugins that load commands work correctly."""

        plugin_name = "plugin"
        if sys.platform.startswith("darwin"):
            plugin_lib_name = "lib%s.dylib" % plugin_name
        else:
            plugin_lib_name = "lib%s.so" % plugin_name

        # Invoke the library build rule.
        self.buildLibrary("plugin.cpp", plugin_name)

        debugger = lldb.SBDebugger.Create()

        retobj = lldb.SBCommandReturnObject()

        retval = debugger.GetCommandInterpreter().HandleCommand("plugin load %s" % plugin_lib_name, retobj)

        retobj.Clear()

        retval = debugger.GetCommandInterpreter().HandleCommand("plugin_loaded_command child abc def ghi",retobj)

        if self.TraceOn():
            print retobj.GetOutput()

        self.expect(retobj,substrs = ['abc def ghi'], exe=False)

        retobj.Clear()

        # check that abbreviations work correctly in plugin commands.
        retval = debugger.GetCommandInterpreter().HandleCommand("plugin_loaded_ ch abc def ghi",retobj)

        if self.TraceOn():
            print retobj.GetOutput()

        self.expect(retobj,substrs = ['abc def ghi'], exe=False)


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
