"""
Test some lldb command abbreviations.
"""

import os, time
import unittest2
import lldb
from lldbtest import *

class AbbreviationsTestCase(TestBase):
    
    mydir = "abbreviation_tests"

    def test_nonrunning_command_abbreviations (self):
        self.expect("ap script",
                    startstr = "The following commands may relate to 'script':",
                    substrs = ['breakpoint command add',
                               'breakpoint command list',
                               'breakpoint list',
                               'command alias',
                               'expression',
                               'script'])

        self.runCmd("com a alias com al")
        self.runCmd("alias gurp help")
        self.expect("gurp target create",
                    substrs = ['Syntax: target create <cmd-options> <filename>'])
        self.runCmd("com u gurp")
        self.expect("gurp",
                    COMMAND_FAILED_AS_EXPECTED, error = True,
                    substrs = ["error: 'gurp' is not a valid command."])

        self.expect("h",
                    startstr = "The following is a list of built-in, permanent debugger commands:")


        self.expect("com s ./change_prompt.lldb",
                    patterns = ["Executing commands in '.*change_prompt.lldb'"])

        self.expect("settings show prompt",
                    startstr = 'prompt (string) = "[old-oak]"')


        self.runCmd("settings set -r prompt")
        self.expect("settings show prompt",
                    startstr = 'prompt (string) = "(lldb) "')


        self.expect("lo li",
                    startstr = "Logging categories for ")

        self.runCmd("se se prompt Sycamore> ")
        self.expect("se sh prompt",
                    startstr = 'prompt (string) = "Sycamore>"')

        self.runCmd("se se -r prompt")
        self.expect("set sh prompt",
                    startstr = 'prompt (string) = "(lldb) "')

        # We don't want to display the stdout if not in TraceOn() mode.
        if not self.TraceOn():
            self.HideStdout()

        self.runCmd (r'''sc print "\n\n\tHello!\n"''')


    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_with_dsym (self):
        self.buildDsym ()
        self.running_abbreviations ()

    def test_with_dwarf (self):
        self.buildDwarf ()
        self.running_abbreviations ()

    def running_abbreviations (self):
        exe = os.path.join (os.getcwd(), "a.out")
        self.expect("fil " + exe,
                    patterns = [ "Current executable set to .*a.out.*" ])

        self.expect("_regexp-b product",
                    substrs = [ "breakpoint set --name 'product'",
                                "Breakpoint created: 1: name = 'product', locations = 1" ])

        self.expect("br s -n sum",
                    startstr = "Breakpoint created: 2: name = 'sum', locations = 1")

        self.expect("br s -f main.cpp -l 32",
                    startstr = "Breakpoint created: 3: file ='main.cpp', line = 32, locations = 1")

        self.runCmd("br co a -s python 1 -o 'print frame'")
        self.expect("br co l 1",
                    substrs = [ "Breakpoint 1:",
                                "Breakpoint commands:",
                                "print frame" ])

        self.runCmd("br co rem 1")
        self.expect("breakpoint command list 1",
                    startstr = "Breakpoint 1 does not have an associated command.")

        self.expect("br di",
                    startstr = 'All breakpoints disabled. (3 breakpoints)')

        self.expect("bre e",
                    startstr = "All breakpoints enabled. (3 breakpoints)")

        self.expect("break list",
                    substrs = ["1: name = 'product', locations = 1",
                               "2: name = 'sum', locations = 1",
                               "3: file ='main.cpp', line = 32, locations = 1"])
        self.expect("br cl -l 32 -f main.cpp",
                    startstr = "1 breakpoints cleared:",
                    substrs = ["3: file ='main.cpp', line = 32, locations = 1"])

        self.expect("pro la",
                    patterns = [ "Process .* launched: "])

        self.expect("pro st",
                    patterns = [ "Process .* stopped",
                                 "thread #1:",
                                 "a.out",
                                 "sum\(int, int\)",
                                 "at main.cpp\:25", 
                                 "stop reason = breakpoint 2.1" ])

        self.expect("dis -f",
                    startstr = "a.out`sum(int, int):",
                    substrs = [' push',
                               ' mov',
                               ' addl ',
                               'ret'],
                    patterns = ['(leave|popq|popl)'])                               

        self.expect("i d l main.cpp",
                    patterns = ["Line table for .*main.cpp in `a.out"])

        self.expect("i d se",
                    patterns = ["Dumping sections for [0-9]+ modules."])

        self.expect("i d symf",
                    patterns = ["Dumping debug symbols for [0-9]+ modules."])

        self.expect("i d symt",
                    patterns = ["Dumping symbol table for [0-9]+ modules."])

        self.expect("i li",
                    substrs = [ 'a.out',
                                '/usr/lib/dyld',
                                '/usr/lib/libstdc++',
                                '/usr/lib/libSystem.B.dylib',
                                '/usr/lib/system/libmathCommon.A.dylib'])

        # Terminate the current process being debugged.
        # The test framework was relying on detecting either "run" or
        # "process launch" command to automatically kill the inferior.
        self.runCmd("process kill")


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()

