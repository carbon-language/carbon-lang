"""
Use lldb Python API to test dynamic values in C++
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class DynamicValueTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

        # Find the line number to break for main.c.

        self.do_something_line = line_number(
            'pass-to-base.cpp', '// Break here in doSomething.')
        self.main_first_call_line = line_number(
            'pass-to-base.cpp',
            '// Break here and get real addresses of myB and otherB.')
        self.main_second_call_line = line_number(
            'pass-to-base.cpp', '// Break here and get real address of reallyA.')

    @add_test_categories(['pyapi'])
    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24663")
    def test_get_dynamic_vals(self):
        """Test fetching C++ dynamic values from pointers & references."""
        self.build()
        exe = self.getBuildArtifact("a.out")

        # Create a target from the debugger.

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Set up our breakpoints:

        do_something_bpt = target.BreakpointCreateByLocation(
            'pass-to-base.cpp', self.do_something_line)
        self.assertTrue(do_something_bpt,
                        VALID_BREAKPOINT)

        first_call_bpt = target.BreakpointCreateByLocation(
            'pass-to-base.cpp', self.main_first_call_line)
        self.assertTrue(first_call_bpt,
                        VALID_BREAKPOINT)

        second_call_bpt = target.BreakpointCreateByLocation(
            'pass-to-base.cpp', self.main_second_call_line)
        self.assertTrue(second_call_bpt,
                        VALID_BREAKPOINT)

        # Now launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())

        self.assertState(process.GetState(), lldb.eStateStopped,
                         PROCESS_STOPPED)

        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, first_call_bpt)
        self.assertEquals(len(threads), 1)
        thread = threads[0]

        frame = thread.GetFrameAtIndex(0)

        # Now find the dynamic addresses of myB and otherB so we can compare them
        # with the dynamic values we get in doSomething:

        use_dynamic = lldb.eDynamicCanRunTarget
        no_dynamic = lldb.eNoDynamicValues

        myB = frame.FindVariable('myB', no_dynamic)
        self.assertTrue(myB)
        myB_loc = int(myB.GetLocation(), 16)

        otherB = frame.FindVariable('otherB', no_dynamic)
        self.assertTrue(otherB)
        otherB_loc = int(otherB.GetLocation(), 16)

        # Okay now run to doSomething:

        threads = lldbutil.continue_to_breakpoint(process, do_something_bpt)
        self.assertEquals(len(threads), 1)
        thread = threads[0]

        frame = thread.GetFrameAtIndex(0)

        # Get "this" using FindVariable:

        this_static = frame.FindVariable('this', no_dynamic)
        this_dynamic = frame.FindVariable('this', use_dynamic)
        self.examine_value_object_of_this_ptr(
            this_static, this_dynamic, myB_loc)

        # Now make sure that the "GetDynamicValue" works:
        # This doesn't work currently because we can't get dynamic values from
        # ConstResult objects.
        fetched_dynamic_value = this_static.GetDynamicValue(use_dynamic)
        self.examine_value_object_of_this_ptr(
            this_static, fetched_dynamic_value, myB_loc)

        # And conversely that the GetDynamicValue() interface also works:
        fetched_static_value = this_dynamic.GetStaticValue()
        self.examine_value_object_of_this_ptr(
            fetched_static_value, this_dynamic, myB_loc)

        # Get "this" using FindValue, make sure that works too:
        this_static = frame.FindValue(
            'this', lldb.eValueTypeVariableArgument, no_dynamic)
        this_dynamic = frame.FindValue(
            'this', lldb.eValueTypeVariableArgument, use_dynamic)
        self.examine_value_object_of_this_ptr(
            this_static, this_dynamic, myB_loc)

        # Get "this" using the EvaluateExpression:
        this_static = frame.EvaluateExpression('this', False)
        this_dynamic = frame.EvaluateExpression('this', True)
        self.examine_value_object_of_this_ptr(
            this_static, this_dynamic, myB_loc)

        # The "frame var" code uses another path to get into children, so let's
        # make sure that works as well:

        self.expect(
            'frame var -d run-target --ptr-depth=2 --show-types anotherA.m_client_A',
            'frame var finds its way into a child member',
            patterns=['\(B \*\)'])

        # Now make sure we also get it right for a reference as well:

        anotherA_static = frame.FindVariable('anotherA', False)
        self.assertTrue(anotherA_static)
        anotherA_static_addr = int(anotherA_static.GetValue(), 16)

        anotherA_dynamic = frame.FindVariable('anotherA', True)
        self.assertTrue(anotherA_dynamic)
        anotherA_dynamic_addr = int(anotherA_dynamic.GetValue(), 16)
        anotherA_dynamic_typename = anotherA_dynamic.GetTypeName()
        self.assertNotEqual(anotherA_dynamic_typename.find('B'), -1)

        self.assertTrue(anotherA_dynamic_addr < anotherA_static_addr)

        anotherA_m_b_value_dynamic = anotherA_dynamic.GetChildMemberWithName(
            'm_b_value', True)
        self.assertTrue(anotherA_m_b_value_dynamic)
        anotherA_m_b_val = int(anotherA_m_b_value_dynamic.GetValue(), 10)
        self.assertEquals(anotherA_m_b_val, 300)

        anotherA_m_b_value_static = anotherA_static.GetChildMemberWithName(
            'm_b_value', True)
        self.assertFalse(anotherA_m_b_value_static)

        # Okay, now continue again, and when we hit the second breakpoint in
        # main

        threads = lldbutil.continue_to_breakpoint(process, second_call_bpt)
        self.assertEquals(len(threads), 1)
        thread = threads[0]

        frame = thread.GetFrameAtIndex(0)
        reallyA_value = frame.FindVariable('reallyA', False)
        self.assertTrue(reallyA_value)
        reallyA_loc = int(reallyA_value.GetLocation(), 16)

        # Finally continue to doSomething again, and make sure we get the right value for anotherA,
        # which this time around is just an "A".

        threads = lldbutil.continue_to_breakpoint(process, do_something_bpt)
        self.assertEquals(len(threads), 1)
        thread = threads[0]

        frame = thread.GetFrameAtIndex(0)
        anotherA_value = frame.FindVariable('anotherA', True)
        self.assertTrue(anotherA_value)
        anotherA_loc = int(anotherA_value.GetValue(), 16)
        self.assertEquals(anotherA_loc, reallyA_loc)
        self.assertEquals(anotherA_value.GetTypeName().find('B'), -1)

    def examine_value_object_of_this_ptr(
            self, this_static, this_dynamic, dynamic_location):
        # Get "this" as its static value
        self.assertTrue(this_static)
        this_static_loc = int(this_static.GetValue(), 16)

        # Get "this" as its dynamic value

        self.assertTrue(this_dynamic)
        this_dynamic_typename = this_dynamic.GetTypeName()
        self.assertNotEqual(this_dynamic_typename.find('B'), -1)
        this_dynamic_loc = int(this_dynamic.GetValue(), 16)

        # Make sure we got the right address for "this"

        self.assertEquals(this_dynamic_loc, dynamic_location)

        # And that the static address is greater than the dynamic one

        self.assertTrue(this_static_loc > this_dynamic_loc)

        # Now read m_b_value which is only in the dynamic value:

        use_dynamic = lldb.eDynamicCanRunTarget
        no_dynamic = lldb.eNoDynamicValues

        this_dynamic_m_b_value = this_dynamic.GetChildMemberWithName(
            'm_b_value', use_dynamic)
        self.assertTrue(this_dynamic_m_b_value)

        m_b_value = int(this_dynamic_m_b_value.GetValue(), 0)
        self.assertEquals(m_b_value, 10)

        # Make sure it is not in the static version

        this_static_m_b_value = this_static.GetChildMemberWithName(
            'm_b_value', no_dynamic)
        self.assertFalse(this_static_m_b_value)

        # Okay, now let's make sure that we can get the dynamic type of a child
        # element:

        contained_auto_ptr = this_dynamic.GetChildMemberWithName(
            'm_client_A', use_dynamic)
        self.assertTrue(contained_auto_ptr)
        contained_b = contained_auto_ptr.GetChildMemberWithName(
            '_M_ptr', use_dynamic)
        if not contained_b:
            contained_b = contained_auto_ptr.GetChildMemberWithName(
                '__ptr_', use_dynamic)
        self.assertTrue(contained_b)

        contained_b_static = contained_auto_ptr.GetChildMemberWithName(
            '_M_ptr', no_dynamic)
        if not contained_b_static:
            contained_b_static = contained_auto_ptr.GetChildMemberWithName(
                '__ptr_', no_dynamic)
        self.assertTrue(contained_b_static)

        contained_b_addr = int(contained_b.GetValue(), 16)
        contained_b_static_addr = int(contained_b_static.GetValue(), 16)

        self.assertTrue(contained_b_addr < contained_b_static_addr)
