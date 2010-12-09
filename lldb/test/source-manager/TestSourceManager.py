"""
Test lldb core component: SourceManager.

Test cases:
1. test_modify_source_file_while_debugging:
   Test the caching mechanism of the source manager.
"""

import os, time
import re
import unittest2
import lldb, lldbutil
from lldbtest import *

class SourceManagerTestCase(TestBase):

    mydir = "source-manager"

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number('main.c', '// Set break point at this line.')

    def test_modify_source_file_while_debugging(self):
        """Modify a source file while debugging the executable."""
        self.buildDefault()
        self.modify_source_file_while_debugging()

    def modify_source_file_while_debugging(self):
        """Modify a source file while debugging the executable."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        self.expect("breakpoint set -f main.c -l %d" % self.line,
                    BREAKPOINT_CREATED,
            startstr = "Breakpoint created: 1: file ='main.c', line = %d, locations = 1" %
                        self.line)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['state is stopped',
                       'main.c',
                       'stop reason = breakpoint'])

        # Display some source code.
        self.expect("list -f main.c -l %d" % self.line, SOURCE_DISPLAYED_CORRECTLY,
            substrs = ['Hello world'])

        # Read the main.c file content.
        with open('main.c', 'r') as f:
            original_content = f.read()
            print "original content:", original_content

        # Modify the in-memory copy of the original source code.
        new_content = original_content.replace('Hello world', 'Hello lldb', 1)

        # This is the function to restore the original content.
        def restore_file():
            with open('main.c', 'w') as f:
                f.write(original_content)
            with open('main.c', 'r') as f:
                print "content restored to:", f.read()

        # Modify the source code file.
        with open('main.c', 'w') as f:
            f.write(new_content)
            print "new content:", new_content
            # Add teardown hook to restore the file to the original content.
            self.addTearDownHook(restore_file)

        # Display the source code again.  We should see the updated line.
        self.expect("list -f main.c -l %d" % self.line, SOURCE_DISPLAYED_CORRECTLY,
            substrs = ['Hello lldb'])


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
