"""
Test lldb core component: SourceManager.

Test cases:

o test_display_source_python:
  Test display of source using the SBSourceManager API.
o test_modify_source_file_while_debugging:
  Test the caching mechanism of the source manager.
"""

import unittest2
import lldb
from lldbtest import *

class SourceManagerTestCase(TestBase):

    mydir = "source-manager"

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number('main.c', '// Set break point at this line.')

    @python_api_test
    def test_display_source_python(self):
        """Test display of source using the SBSourceManager API."""
        self.buildDefault()
        self.display_source_python()

    def test_modify_source_file_while_debugging(self):
        """Modify a source file while debugging the executable."""
        self.buildDefault()
        self.modify_source_file_while_debugging()

    def display_source_python(self):
        """Display source using the SBSourceManager API."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target.IsValid(), VALID_TARGET)

        # Launch the process, and do not stop at the entry point.
        error = lldb.SBError()
        process = target.Launch (self.dbg.GetListener(), None, None, os.ctermid(), os.ctermid(), os.ctermid(), None, 0, False, error)

        #
        # Exercise Python APIs to display source lines.
        #

        # Create the filespec for 'main.c'.
        filespec = lldb.SBFileSpec('main.c', False)
        source_mgr = self.dbg.GetSourceManager()
        # Use a string stream as the destination.
        stream = lldb.SBStream()
        source_mgr.DisplaySourceLinesWithLineNumbers(filespec,
                                                     self.line,
                                                     2, # context before
                                                     2, # context after
                                                     "=>", # prefix for current line
                                                     stream)

        #    2   	
        #    3    int main(int argc, char const *argv[]) {
        # => 4        printf("Hello world.\n"); // Set break point at this line.
        #    5        return 0;
        #    6    }
        self.expect(stream.GetData(), "Source code displayed correctly",
                    exe=False,
            patterns = ['=> %d.*Hello world' % self.line])        

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
            substrs = ['stopped',
                       'main.c',
                       'stop reason = breakpoint'])

        # Display some source code.
        self.expect("list -f main.c -l %d" % self.line, SOURCE_DISPLAYED_CORRECTLY,
            substrs = ['Hello world'])

        # Read the main.c file content.
        with open('main.c', 'r') as f:
            original_content = f.read()
            if self.TraceOn():
                print "original content:", original_content

        # Modify the in-memory copy of the original source code.
        new_content = original_content.replace('Hello world', 'Hello lldb', 1)

        # This is the function to restore the original content.
        def restore_file():
            #print "os.path.getmtime() before restore:", os.path.getmtime('main.c')
            time.sleep(1)
            with open('main.c', 'w') as f:
                f.write(original_content)
            if self.TraceOn():
                with open('main.c', 'r') as f:
                    print "content restored to:", f.read()
            # Touch the file just to be sure.
            os.utime('main.c', None)
            if self.TraceOn():
                print "os.path.getmtime() after restore:", os.path.getmtime('main.c')



        # Modify the source code file.
        with open('main.c', 'w') as f:
            time.sleep(1)
            f.write(new_content)
            if self.TraceOn():
                print "new content:", new_content
                print "os.path.getmtime() after writing new content:", os.path.getmtime('main.c')
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
