"""
Test Debugger APIs.
"""

import os, time
import re
import unittest2
import lldb, lldbutil
from lldbtest import *

class DebuggerAPITestCase(TestBase):

    mydir = os.path.join("python_api", "debugger")

    @python_api_test
    def test_debugger_api_boundary_condition(self):
        """Exercise SBDebugger APIs with boundary conditions."""
        self.dbg.HandleCommand(None)
        self.dbg.SetDefaultArchitecture(None)
        self.dbg.GetScriptingLanguage(None)
        self.dbg.CreateTarget(None)
        self.dbg.CreateTarget(None, None, None, True, lldb.SBError())
        self.dbg.CreateTargetWithFileAndTargetTriple(None, None)
        self.dbg.CreateTargetWithFileAndArch(None, None)
        self.dbg.FindTargetWithFileAndArch(None, None)
        self.dbg.SetInternalVariable(None, None, None)
        self.dbg.GetInternalVariableValue(None, None)
        self.dbg.SetPrompt(None)
        self.dbg.SetCurrentPlatform(None)
        self.dbg.SetCurrentPlatformSDKRoot(None)
