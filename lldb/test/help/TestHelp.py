"""
Test some lldb help commands.

See also CommandInterpreter::OutputFormattedHelpText().
"""

import os, time
import unittest2
import lldb
from lldbtest import *

class HelpCommandTestCase(TestBase):

    mydir = "help"

    def test_simplehelp(self):
        """A simple test of 'help' command and its output."""
        self.expect("help",
            startstr = 'The following is a list of built-in, permanent debugger commands')

    def version_number_string(self):
        """Helper function to find the version number string of lldb."""
        plist = os.path.join(os.getcwd(), os.pardir, os.pardir, "resources", "LLDB-info.plist")
        try:
            CFBundleVersionSegFound = False
            with open(plist, 'r') as f:
                for line in f:
                    if CFBundleVersionSegFound:
                        version_line = line.strip()
                        import re
                        m = re.match("<string>(.*)</string>", version_line)
                        if m:
                            version = m.group(1)
                            return version
                        else:
                            # Unsuccessful, let's juts break out of the for loop.
                            break

                    if line.find("<key>CFBundleVersion</key>") != -1:
                        # Found our match.  The next line contains our version
                        # string, for example:
                        # 
                        #     <string>38</string>
                        CFBundleVersionSegFound = True

        except:
            # Just fallthrough...
            print "Unexpected error:", sys.exc_info()[0]
            pass

        # Use None to signify that we are not able to grok the version number.
        return None


    def test_help_version(self):
        """Test 'help version' and 'version' commands."""
        self.expect("help version",
            substrs = ['Show version of LLDB debugger.'])
        version_str = self.version_number_string()
        self.expect("version",
            patterns = ['LLDB-' + (version_str if version_str else '[0-9]+')])

    def test_help_should_not_hang_emacsshell(self):
        """Command 'settings set term-width 0' should not hang the help command."""
        self.runCmd("settings set term-width 0")
        self.expect("help",
            startstr = 'The following is a list of built-in, permanent debugger commands')

    def test_help_image_dump_symtab_should_not_crash(self):
        """Command 'help image dump symtab' should not crash lldb."""
        self.expect("help image dump symtab",
            substrs = ['image dump symtab',
                       'sort-order'])

    def test_help_image_du_sym_is_ambiguous(self):
        """Command 'help image du sym' is ambiguous and spits out the list of candidates."""
        self.expect("help image du sym",
                    COMMAND_FAILED_AS_EXPECTED, error=True,
            substrs = ['error: ambiguous command image du sym',
                       'symfile',
                       'symtab'])

    def test_help_image_du_line_should_work(self):
        """Command 'help image du line' is not ambiguous and should work."""
        self.expect("help image du line",
            substrs = ['Dump the debug symbol file for one or more executable images'])


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
