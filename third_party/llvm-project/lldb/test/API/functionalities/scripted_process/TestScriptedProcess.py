"""
Test python scripted process in lldb
"""

import os

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test import lldbtest

class ScriptedProcesTestCase(TestBase):

    NO_DEBUG_INFO_TESTCASE = True

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        TestBase.setUp(self)

    def tearDown(self):
        TestBase.tearDown(self)

    def test_python_plugin_package(self):
        """Test that the lldb python module has a `plugins.scripted_process`
        package."""
        self.expect('script import lldb.plugins',
                    substrs=["ModuleNotFoundError"], matching=False)

        self.expect('script dir(lldb.plugins)',
                    substrs=["scripted_process"])

        self.expect('script import lldb.plugins.scripted_process',
                    substrs=["ModuleNotFoundError"], matching=False)

        self.expect('script dir(lldb.plugins.scripted_process)',
                    substrs=["ScriptedProcess"])

        self.expect('script from lldb.plugins.scripted_process import ScriptedProcess',
                    substrs=["ImportError"], matching=False)

        self.expect('script dir(ScriptedProcess)',
                    substrs=["launch"])

    @skipUnlessDarwin
    def test_invalid_scripted_register_context(self):
        """Test that we can launch an lldb scripted process with an invalid
        Scripted Thread, with invalid register context."""
        self.build()
        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(target, VALID_TARGET)
        log_file = self.getBuildArtifact('thread.log')
        self.runCmd("log enable lldb thread -f " + log_file)
        self.assertTrue(os.path.isfile(log_file))

        os.environ['SKIP_SCRIPTED_PROCESS_LAUNCH'] = '1'
        def cleanup():
          del os.environ["SKIP_SCRIPTED_PROCESS_LAUNCH"]
        self.addTearDownHook(cleanup)

        scripted_process_example_relpath = 'invalid_scripted_process.py'
        self.runCmd("command script import " + os.path.join(self.getSourceDir(),
                                                            scripted_process_example_relpath))

        launch_info = lldb.SBLaunchInfo(None)
        launch_info.SetProcessPluginName("ScriptedProcess")
        launch_info.SetScriptedProcessClassName("invalid_scripted_process.InvalidScriptedProcess")
        error = lldb.SBError()

        process = target.Launch(launch_info, error)

        self.assertSuccess(error)
        self.assertTrue(process, PROCESS_IS_VALID)
        self.assertEqual(process.GetProcessID(), 666)
        self.assertEqual(process.GetNumThreads(), 0)

        with open(log_file, 'r') as f:
            log = f.read()

        self.assertIn("Failed to get scripted thread registers data.", log)

    @skipIf(archs=no_match(['x86_64', 'arm64', 'arm64e']))
    def test_scripted_process_and_scripted_thread(self):
        """Test that we can launch an lldb scripted process using the SBAPI,
        check its process ID, read string from memory, check scripted thread
        id, name stop reason and register context.
        """
        self.build()
        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(target, VALID_TARGET)

        os.environ['SKIP_SCRIPTED_PROCESS_LAUNCH'] = '1'
        def cleanup():
          del os.environ["SKIP_SCRIPTED_PROCESS_LAUNCH"]
        self.addTearDownHook(cleanup)

        scripted_process_example_relpath = 'dummy_scripted_process.py'
        self.runCmd("command script import " + os.path.join(self.getSourceDir(),
                                                            scripted_process_example_relpath))

        launch_info = lldb.SBLaunchInfo(None)
        launch_info.SetProcessPluginName("ScriptedProcess")
        launch_info.SetScriptedProcessClassName("dummy_scripted_process.DummyScriptedProcess")

        error = lldb.SBError()
        process = target.Launch(launch_info, error)
        self.assertTrue(process and process.IsValid(), PROCESS_IS_VALID)
        self.assertEqual(process.GetProcessID(), 42)

        self.assertEqual(process.GetNumThreads(), 1)

        thread = process.GetSelectedThread()
        self.assertTrue(thread, "Invalid thread.")
        self.assertEqual(thread.GetThreadID(), 0x19)
        self.assertEqual(thread.GetName(), "DummyScriptedThread.thread-1")
        self.assertEqual(thread.GetStopReason(), lldb.eStopReasonSignal)

        self.assertGreater(thread.GetNumFrames(), 0)

        frame = thread.GetFrameAtIndex(0)
        GPRs = None
        register_set = frame.registers # Returns an SBValueList.
        for regs in register_set:
            if 'general purpose' in regs.name.lower():
                GPRs = regs
                break

        self.assertTrue(GPRs, "Invalid General Purpose Registers Set")
        self.assertGreater(GPRs.GetNumChildren(), 0)
        for idx, reg in enumerate(GPRs, start=1):
            if idx > 21:
                break
            self.assertEqual(idx, int(reg.value, 16))

        self.assertTrue(frame.IsArtificial(), "Frame is not artificial")
        pc = frame.GetPCAddress().GetLoadAddress(target)
        self.assertEqual(pc, 0x0100001b00)
