"""
Test breakpoint serialization.
"""

import os
import json
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class BreakpointSerialization(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    @add_test_categories(['pyapi'])
    def test_resolvers(self):
        """Use Python APIs to test that we serialize resolvers."""
        self.build()
        self.setup_targets_and_cleanup()
        self.do_check_resolvers()

    def test_filters(self):
        """Use Python APIs to test that we serialize search filters correctly."""
        self.build()
        self.setup_targets_and_cleanup()
        self.do_check_filters()

    def test_options(self):
        """Use Python APIs to test that we serialize breakpoint options correctly."""
        self.build()
        self.setup_targets_and_cleanup()
        self.do_check_options()

    def test_appending(self):
        """Use Python APIs to test that we serialize breakpoint options correctly."""
        self.build()
        self.setup_targets_and_cleanup()
        self.do_check_appending()

    def test_name_filters(self):
        """Use python APIs to test that reading in by name works correctly."""
        self.build()
        self.setup_targets_and_cleanup()
        self.do_check_names()

    def test_scripted_extra_args(self):
        self.build()
        self.setup_targets_and_cleanup()
        self.do_check_extra_args()

    def test_structured_data_serialization(self):
        target = self.dbg.GetDummyTarget()
        self.assertTrue(target.IsValid(), VALID_TARGET)

        interpreter = self.dbg.GetCommandInterpreter()
        result = lldb.SBCommandReturnObject()
        interpreter.HandleCommand("br set -f foo -l 42", result)
        result = lldb.SBCommandReturnObject()
        interpreter.HandleCommand("br set -c 'argc == 1' -n main", result)

        bkp1 =  target.GetBreakpointAtIndex(0)
        self.assertTrue(bkp1.IsValid(), VALID_BREAKPOINT)
        stream = lldb.SBStream()
        sd = bkp1.SerializeToStructuredData()
        sd.GetAsJSON(stream)
        serialized_data = json.loads(stream.GetData())
        self.assertEqual(serialized_data["Breakpoint"]["BKPTResolver"]["Options"]["FileName"], "foo")
        self.assertEqual(serialized_data["Breakpoint"]["BKPTResolver"]["Options"]["LineNumber"], 42)

        bkp2 =  target.GetBreakpointAtIndex(1)
        self.assertTrue(bkp2.IsValid(), VALID_BREAKPOINT)
        stream = lldb.SBStream()
        sd = bkp2.SerializeToStructuredData()
        sd.GetAsJSON(stream)
        serialized_data = json.loads(stream.GetData())
        self.assertIn("main", serialized_data["Breakpoint"]["BKPTResolver"]["Options"]["SymbolNames"])
        self.assertEqual(serialized_data["Breakpoint"]["BKPTOptions"]["ConditionText"],"argc == 1")

        invalid_bkp = lldb.SBBreakpoint()
        self.assertFalse(invalid_bkp.IsValid(), "Breakpoint should not be valid.")
        stream = lldb.SBStream()
        sd = invalid_bkp.SerializeToStructuredData()
        sd.GetAsJSON(stream)
        self.assertFalse(stream.GetData(), "Invalid breakpoint should have an empty structured data")

    def setup_targets_and_cleanup(self):
        def cleanup ():
            self.RemoveTempFile(self.bkpts_file_path)

            if self.orig_target.IsValid():
                self.dbg.DeleteTarget(self.orig_target)
                self.dbg.DeleteTarget(self.copy_target)

        self.addTearDownHook(cleanup)
        self.RemoveTempFile(self.bkpts_file_path)

        exe = self.getBuildArtifact("a.out")

        # Create the targets we are making breakpoints in and copying them to:
        self.orig_target = self.dbg.CreateTarget(exe)
        self.assertTrue(self.orig_target, VALID_TARGET)

        self.copy_target = self.dbg.CreateTarget(exe)
        self.assertTrue(self.copy_target, VALID_TARGET)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

        self.bkpts_file_path = self.getBuildArtifact("breakpoints.json")
        self.bkpts_file_spec = lldb.SBFileSpec(self.bkpts_file_path)

    def check_equivalence(self, source_bps, do_write = True):

        error = lldb.SBError()

        if (do_write):
            error = self.orig_target.BreakpointsWriteToFile(self.bkpts_file_spec, source_bps)
            self.assertTrue(error.Success(), "Failed writing breakpoints to file: %s."%(error.GetCString()))

        copy_bps = lldb.SBBreakpointList(self.copy_target)
        error = self.copy_target.BreakpointsCreateFromFile(self.bkpts_file_spec, copy_bps)
        self.assertTrue(error.Success(), "Failed reading breakpoints from file: %s"%(error.GetCString()))

        num_source_bps = source_bps.GetSize()
        num_copy_bps = copy_bps.GetSize()
        self.assertEqual(num_source_bps, num_copy_bps, "Didn't get same number of input and output breakpoints - orig: %d copy: %d"%(num_source_bps, num_copy_bps))

        for i in range(0, num_source_bps):
            source_bp = source_bps.GetBreakpointAtIndex(i)
            source_desc = lldb.SBStream()
            source_bp.GetDescription(source_desc, False)
            source_text = source_desc.GetData()

            # I am assuming here that the breakpoints will get written out in breakpoint ID order, and
            # read back in ditto.  That is true right now, and I can't see any reason to do it differently
            # but if we do we can go to writing the breakpoints one by one, or sniffing the descriptions to
            # see which one is which.
            copy_id = source_bp.GetID()
            copy_bp = copy_bps.FindBreakpointByID(copy_id)
            self.assertTrue(copy_bp.IsValid(), "Could not find copy breakpoint %d."%(copy_id))

            copy_desc = lldb.SBStream()
            copy_bp.GetDescription(copy_desc, False)
            copy_text = copy_desc.GetData()

            # These two should be identical.
            self.trace("Source text for %d is %s."%(i, source_text))
            self.assertTrue (source_text == copy_text, "Source and dest breakpoints are not identical: \nsource: %s\ndest: %s"%(source_text, copy_text))

    def do_check_resolvers(self):
        """Use Python APIs to check serialization of breakpoint resolvers"""

        empty_module_list = lldb.SBFileSpecList()
        empty_cu_list = lldb.SBFileSpecList()
        blubby_file_spec = lldb.SBFileSpec(os.path.join(self.getSourceDir(), "blubby.c"))

        # It isn't actually important for these purposes that these breakpoint
        # actually have locations.
        source_bps = lldb.SBBreakpointList(self.orig_target)
        source_bps.Append(self.orig_target.BreakpointCreateByLocation("blubby.c", 666))
        # Make sure we do one breakpoint right:
        self.check_equivalence(source_bps)
        source_bps.Clear()

        source_bps.Append(self.orig_target.BreakpointCreateByName("blubby", lldb.eFunctionNameTypeAuto, empty_module_list, empty_cu_list))
        source_bps.Append(self.orig_target.BreakpointCreateByName("blubby", lldb.eFunctionNameTypeFull, empty_module_list,empty_cu_list))
        source_bps.Append(self.orig_target.BreakpointCreateBySourceRegex("dont really care", blubby_file_spec))

        # And some number greater than one:
        self.check_equivalence(source_bps)

    def do_check_filters(self):
        """Use Python APIs to check serialization of breakpoint filters."""
        module_list = lldb.SBFileSpecList()
        module_list.Append(lldb.SBFileSpec("SomeBinary"))
        module_list.Append(lldb.SBFileSpec("SomeOtherBinary"))

        cu_list = lldb.SBFileSpecList()
        cu_list.Append(lldb.SBFileSpec("SomeCU.c"))
        cu_list.Append(lldb.SBFileSpec("AnotherCU.c"))
        cu_list.Append(lldb.SBFileSpec("ThirdCU.c"))

        blubby_file_spec = lldb.SBFileSpec(os.path.join(self.getSourceDir(), "blubby.c"))

        # It isn't actually important for these purposes that these breakpoint
        # actually have locations.
        source_bps = lldb.SBBreakpointList(self.orig_target)
        bkpt = self.orig_target.BreakpointCreateByLocation(blubby_file_spec, 666, 0, module_list)
        source_bps.Append(bkpt)

        # Make sure we do one right:
        self.check_equivalence(source_bps)
        source_bps.Clear()

        bkpt = self.orig_target.BreakpointCreateByName("blubby", lldb.eFunctionNameTypeAuto, module_list, cu_list)
        source_bps.Append(bkpt)
        bkpt = self.orig_target.BreakpointCreateByName("blubby", lldb.eFunctionNameTypeFull, module_list, cu_list)
        source_bps.Append(bkpt)
        bkpt = self.orig_target.BreakpointCreateBySourceRegex("dont really care", blubby_file_spec)
        source_bps.Append(bkpt)

        # And some number greater than one:
        self.check_equivalence(source_bps)

    def do_check_options(self):
        """Use Python APIs to check serialization of breakpoint options."""

        empty_module_list = lldb.SBFileSpecList()
        empty_cu_list = lldb.SBFileSpecList()
        blubby_file_spec = lldb.SBFileSpec(os.path.join(self.getSourceDir(), "blubby.c"))

        # It isn't actually important for these purposes that these breakpoint
        # actually have locations.
        source_bps = lldb.SBBreakpointList(self.orig_target)

        bkpt = self.orig_target.BreakpointCreateByLocation(
            lldb.SBFileSpec("blubby.c"), 666, 333, 0, lldb.SBFileSpecList())
        bkpt.SetEnabled(False)
        bkpt.SetOneShot(True)
        bkpt.SetThreadID(10)
        source_bps.Append(bkpt)

        # Make sure we get one right:
        self.check_equivalence(source_bps)
        source_bps.Clear()

        bkpt = self.orig_target.BreakpointCreateByName("blubby", lldb.eFunctionNameTypeAuto, empty_module_list, empty_cu_list)
        bkpt.SetIgnoreCount(10)
        bkpt.SetThreadName("grubby")
        source_bps.Append(bkpt)

        bkpt = self.orig_target.BreakpointCreateByName("blubby", lldb.eFunctionNameTypeAuto, empty_module_list, empty_cu_list)
        bkpt.SetCondition("gonna remove this")
        bkpt.SetCondition("")
        source_bps.Append(bkpt)

        bkpt = self.orig_target.BreakpointCreateByName("blubby", lldb.eFunctionNameTypeFull, empty_module_list,empty_cu_list)
        bkpt.SetCondition("something != something_else")
        bkpt.SetQueueName("grubby")
        bkpt.AddName("FirstName")
        bkpt.AddName("SecondName")
        bkpt.SetScriptCallbackBody('\tprint("I am a function that prints.")\n\tprint("I don\'t do anything else")\n')
        source_bps.Append(bkpt)

        bkpt = self.orig_target.BreakpointCreateBySourceRegex("dont really care", blubby_file_spec)
        cmd_list = lldb.SBStringList()
        cmd_list.AppendString("frame var")
        cmd_list.AppendString("thread backtrace")

        bkpt.SetCommandLineCommands(cmd_list)
        source_bps.Append(bkpt)

        self.check_equivalence(source_bps)

    def do_check_appending(self):
        """Use Python APIs to check appending to already serialized options."""

        empty_module_list = lldb.SBFileSpecList()
        empty_cu_list = lldb.SBFileSpecList()
        blubby_file_spec = lldb.SBFileSpec(os.path.join(self.getSourceDir(), "blubby.c"))

        # It isn't actually important for these purposes that these breakpoint
        # actually have locations.

        all_bps = lldb.SBBreakpointList(self.orig_target)
        source_bps = lldb.SBBreakpointList(self.orig_target)

        bkpt = self.orig_target.BreakpointCreateByLocation(
            lldb.SBFileSpec("blubby.c"), 666, 333, 0, lldb.SBFileSpecList())
        bkpt.SetEnabled(False)
        bkpt.SetOneShot(True)
        bkpt.SetThreadID(10)
        source_bps.Append(bkpt)
        all_bps.Append(bkpt)

        error = lldb.SBError()
        error = self.orig_target.BreakpointsWriteToFile(self.bkpts_file_spec, source_bps)
        self.assertTrue(error.Success(), "Failed writing breakpoints to file: %s."%(error.GetCString()))

        source_bps.Clear()

        bkpt = self.orig_target.BreakpointCreateByName("blubby", lldb.eFunctionNameTypeAuto, empty_module_list, empty_cu_list)
        bkpt.SetIgnoreCount(10)
        bkpt.SetThreadName("grubby")
        source_bps.Append(bkpt)
        all_bps.Append(bkpt)

        bkpt = self.orig_target.BreakpointCreateByName("blubby", lldb.eFunctionNameTypeFull, empty_module_list,empty_cu_list)
        bkpt.SetCondition("something != something_else")
        bkpt.SetQueueName("grubby")
        bkpt.AddName("FirstName")
        bkpt.AddName("SecondName")

        source_bps.Append(bkpt)
        all_bps.Append(bkpt)

        error = self.orig_target.BreakpointsWriteToFile(self.bkpts_file_spec, source_bps, True)
        self.assertTrue(error.Success(), "Failed appending breakpoints to file: %s."%(error.GetCString()))

        self.check_equivalence(all_bps)

    def do_check_names(self):
        bkpt = self.orig_target.BreakpointCreateByLocation(
            lldb.SBFileSpec("blubby.c"), 666, 333, 0, lldb.SBFileSpecList())
        good_bkpt_name = "GoodBreakpoint"
        write_bps = lldb.SBBreakpointList(self.orig_target)
        bkpt.AddName(good_bkpt_name)
        write_bps.Append(bkpt)

        error = lldb.SBError()
        error = self.orig_target.BreakpointsWriteToFile(self.bkpts_file_spec, write_bps)
        self.assertTrue(error.Success(), "Failed writing breakpoints to file: %s."%(error.GetCString()))

        copy_bps = lldb.SBBreakpointList(self.copy_target)
        names_list = lldb.SBStringList()
        names_list.AppendString("NoSuchName")

        error = self.copy_target.BreakpointsCreateFromFile(self.bkpts_file_spec, names_list, copy_bps)
        self.assertTrue(error.Success(), "Failed reading breakpoints from file: %s"%(error.GetCString()))
        self.assertEqual(copy_bps.GetSize(), 0, "Found breakpoints with a nonexistent name.")

        names_list.AppendString(good_bkpt_name)
        error = self.copy_target.BreakpointsCreateFromFile(self.bkpts_file_spec, names_list, copy_bps)
        self.assertTrue(error.Success(), "Failed reading breakpoints from file: %s"%(error.GetCString()))
        self.assertEqual(copy_bps.GetSize(), 1, "Found the matching breakpoint.")

    def do_check_extra_args(self):

        import side_effect
        interp = self.dbg.GetCommandInterpreter()
        error = lldb.SBError()

        script_name = os.path.join(self.getSourceDir(), "resolver.py")

        command = "command script import " + script_name
        result = lldb.SBCommandReturnObject()
        interp.HandleCommand(command, result)
        self.assertTrue(result.Succeeded(), "com scr imp failed: %s"%(result.GetError()))

        # First make sure a scripted breakpoint with no args works:
        bkpt = self.orig_target.BreakpointCreateFromScript("resolver.Resolver", lldb.SBStructuredData(),
                                                           lldb.SBFileSpecList(), lldb.SBFileSpecList())
        self.assertTrue(bkpt.IsValid(), "Bkpt is valid")
        write_bps = lldb.SBBreakpointList(self.orig_target)

        error = self.orig_target.BreakpointsWriteToFile(self.bkpts_file_spec, write_bps)
        self.assertTrue(error.Success(), "Failed writing breakpoints: %s"%(error.GetCString()))

        side_effect.g_extra_args = None
        copy_bps = lldb.SBBreakpointList(self.copy_target)
        error = self.copy_target.BreakpointsCreateFromFile(self.bkpts_file_spec, copy_bps)
        self.assertTrue(error.Success(), "Failed reading breakpoints: %s"%(error.GetCString()))

        self.assertEqual(copy_bps.GetSize(), 1, "Got one breakpoint from file.")
        no_keys = lldb.SBStringList()
        side_effect.g_extra_args.GetKeys(no_keys)
        self.assertEqual(no_keys.GetSize(), 0, "Should have no keys")

        self.orig_target.DeleteAllBreakpoints()
        self.copy_target.DeleteAllBreakpoints()

        # Now try one with extra args:

        extra_args = lldb.SBStructuredData()
        stream = lldb.SBStream()
        stream.Print('{"first_arg" : "first_value", "second_arg" : "second_value"}')
        extra_args.SetFromJSON(stream)
        self.assertTrue(extra_args.IsValid(), "SBStructuredData is valid.")

        bkpt = self.orig_target.BreakpointCreateFromScript("resolver.Resolver",
                                                           extra_args, lldb.SBFileSpecList(), lldb.SBFileSpecList())
        self.assertTrue(bkpt.IsValid(), "Bkpt is valid")
        write_bps = lldb.SBBreakpointList(self.orig_target)

        error = self.orig_target.BreakpointsWriteToFile(self.bkpts_file_spec, write_bps)
        self.assertTrue(error.Success(), "Failed writing breakpoints: %s"%(error.GetCString()))

        orig_extra_args = side_effect.g_extra_args
        self.assertTrue(orig_extra_args.IsValid(), "Extra args originally valid")

        orig_keys = lldb.SBStringList()
        orig_extra_args.GetKeys(orig_keys)
        self.assertEqual(2, orig_keys.GetSize(), "Should have two keys")

        side_effect.g_extra_args = None

        copy_bps = lldb.SBBreakpointList(self.copy_target)
        error = self.copy_target.BreakpointsCreateFromFile(self.bkpts_file_spec, copy_bps)
        self.assertTrue(error.Success(), "Failed reading breakpoints: %s"%(error.GetCString()))

        self.assertEqual(copy_bps.GetSize(), 1, "Got one breakpoint from file.")

        copy_extra_args = side_effect.g_extra_args
        copy_keys = lldb.SBStringList()
        copy_extra_args.GetKeys(copy_keys)
        self.assertEqual(2, copy_keys.GetSize(), "Copy should have two keys")

        for idx in range(0, orig_keys.GetSize()):
            key = orig_keys.GetStringAtIndex(idx)
            copy_value = copy_extra_args.GetValueForKey(key).GetStringValue(100)

            if key == "first_arg":
                self.assertEqual(copy_value, "first_value")
            elif key == "second_arg":
                self.assertEqual(copy_value, "second_value")
            else:
                self.Fail("Unknown key: %s"%(key))
