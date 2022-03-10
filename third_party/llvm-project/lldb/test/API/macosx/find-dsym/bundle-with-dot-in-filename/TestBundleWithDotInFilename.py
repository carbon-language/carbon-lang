"""Test that a dSYM can be found when a binary is in a bundle hnd has dots in the filename."""


#import unittest2
import os.path
from time import sleep

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


exe_name = 'find-bundle-with-dots-in-fn'  # must match Makefile

class BundleWithDotInFilenameTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        TestBase.setUp(self)
        self.source = 'main.c'

    def tearDown(self):
        # Destroy process before TestBase.tearDown()
        self.dbg.GetSelectedTarget().GetProcess().Destroy()

        # Call super's tearDown().
        TestBase.tearDown(self)

    @skipIfRemote
    @skipUnlessDarwin
    # This test is explicitly a dSYM test, it doesn't need to run for any other config.
    @skipIf(debug_info=no_match(["dsym"]))
    def test_attach_and_check_dsyms(self):
        """Test attach to binary, see if the bundle dSYM is found"""
        exe = self.getBuildArtifact(exe_name)
        self.build()
        os.chdir(self.getBuildDir());

        # Use a file as a synchronization point between test and inferior.
        pid_file_path = lldbutil.append_to_process_working_directory(self,
            "token_pid_%d" % (int(os.getpid())))
        self.addTearDownHook(
            lambda: self.run_platform_command(
                "rm %s" %
                (pid_file_path)))

        popen = self.spawnSubprocess(exe, [pid_file_path])

        # Wait for the inferior to start up, dlopen a bundle, remove the bundle it linked in
        pid = lldbutil.wait_for_file_on_target(self, pid_file_path)

        # Since the library that was dlopen()'ed is now removed, lldb will need to find the
        # binary & dSYM via target.exec-search-paths
        settings_str = "settings set target.exec-search-paths " + self.get_process_working_directory() + "/hide.app"
        self.runCmd(settings_str)

        self.runCmd("process attach -p " + str(popen.pid))

        target = self.dbg.GetSelectedTarget()
        self.assertTrue(target.IsValid(), 'Should have a valid Target after attaching to process')

        setup_complete = target.FindFirstGlobalVariable("setup_is_complete")
        self.assertEquals(setup_complete.GetValueAsUnsigned(), 1, 'Check that inferior process has completed setup')

        # Find the bundle module, see if we found the dSYM too (they're both in "hide.app")
        i = 0
        while i < target.GetNumModules():
            mod = target.GetModuleAtIndex(i)
            if mod.GetFileSpec().GetFilename() == 'com.apple.sbd':
                dsym_name = mod.GetSymbolFileSpec().GetFilename()
                self.assertTrue (dsym_name == 'com.apple.sbd', "Check that we found the dSYM for the bundle that was loaded")
            i=i+1
        os.chdir(self.getSourceDir());
