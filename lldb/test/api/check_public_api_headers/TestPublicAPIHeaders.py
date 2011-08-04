"""Test the integrity of the lldb public api directory containing SB*.h headers.

There should be nothing unwanted there and a simpe main.cpp which includes SB*.h
should compile and link with the LLDB framework."""

import os, re, StringIO
import unittest2
from lldbtest import *

class SBDirCheckerCase(TestBase):

    mydir = os.path.join("api", "check_public_api_headers")

    def setUp(self):
        TestBase.setUp(self)
        self.build_dir = os.environ["LLDB_BUILD_DIR"]
        self.template = 'main.cpp.template'
        self.source = 'main.cpp'

    def test_sb_api_directory(self):
        """Test the SB API directory and make sure there's no unwanted stuff."""

        if sys.platform.startswith("darwin") and self.getArchitecture() == "i386":
            self.skipTest("LLDB.framework built 64-bit")

        # Call the program generator to produce main.cpp.
        self.generate_main_cpp()

        d = {'FRAMEWORK_INCLUDES' : "-F%s" % self.build_dir}
        self.buildDefault(dictionary=d)
        self.exe_name = 'a.out'
        self.sanity_check_executable(self.exe_name)

    def generate_main_cpp(self):
        """Generate main.cpp from main.cpp.template."""
        temp = os.path.join(os.getcwd(), self.template)
        with open(temp, 'r') as f:
            content = f.read()

        public_api_dir = os.path.join(os.environ["LLDB_SRC"], "include", "lldb", "API")

        # Look under the include/lldb/API directory and add #include statements
        # for all the SB API headers.
        public_headers = os.listdir(public_api_dir)
        # For different platforms, the include statement can vary.
        if sys.platform.startswith("darwin"):
            include_stmt = "'#include <%s>' % os.path.join('LLDB', header)"
        list = [eval(include_stmt) for header in public_headers if (header.startswith("SB") and
                                                                    header.endswith(".h"))]
        includes = '\n'.join(list)
        new_content = content.replace('%include_SB_APIs%', includes)
        src = os.path.join(os.getcwd(), self.source)
        with open(src, 'w') as f:
            f.write(new_content)

        # The main.cpp has been generated, add a teardown hook to remove it.
        self.addTearDownHook(lambda: os.remove(src))

    def sanity_check_executable(self, exe_name):
        """Sanity check executable compiled from the auto-generated program."""
        exe = os.path.join(os.getcwd(), exe_name)
        self.runCmd("file %s" % exe, CURRENT_EXECUTABLE_SET)

        self.line_to_break = line_number(self.source, '// Set breakpoint here.')

        if sys.platform.startswith("darwin"):
            env_var = 'DYLD_FRAMEWORK_PATH'
            env_val = self.build_dir

        env_cmd = "settings set target.process.env-vars %s=%s" %(env_var, env_val)
        if self.TraceOn():
            print "Set environment to: ", env_cmd
        self.runCmd(env_cmd)
        self.addTearDownHook(lambda: self.runCmd("settings remove target.process.env-vars %s" % env_var))

        self.expect('breakpoint set -f %s -l %d' % (self.source, self.line_to_break),
                    BREAKPOINT_CREATED,
            startstr = "Breakpoint created: 1: file ='%s', line = %d" %
                        (self.source, self.line_to_break))

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        self.runCmd('frame variable')

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
