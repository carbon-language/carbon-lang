"""
Test some lldb command abbreviations to make sure the common short spellings of
many commands remain available even after we add/delete commands in the future.
"""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

class CommonShortSpellingsTestCase(TestBase):
    
    mydir = TestBase.compute_mydir(__file__)

    def test_abbrevs2 (self):
        command_interpreter = self.dbg.GetCommandInterpreter()
        self.assertTrue(command_interpreter, VALID_COMMAND_INTERPRETER)
        result = lldb.SBCommandReturnObject()

        abbrevs = [
            ('br s', 'breakpoint set'),
            ('disp', '_regexp-display'),  # a.k.a., 'display'
            ('di', 'disassemble'),
            ('dis', 'disassemble'),
            ('ta st a', 'target stop-hook add'),
            ('fr v', 'frame variable'),
            ('ta st li', 'target stop-hook list'),
        ]

        for (short, long) in abbrevs:
            command_interpreter.ResolveCommand(short, result)
            self.assertTrue(result.Succeeded())
            self.assertEqual(long, result.GetOutput())


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()

