"""Check that compiler-generated register values work correctly"""

import re
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


def re_expr_equals(val_type, val):
    # Match ({val_type}) ${sum_digits} = {val}
    return re.compile(r'\(' + val_type + '\) \$\d+ = ' + str(val))


class RegisterVariableTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @expectedFailureAll(compiler="clang", compiler_version=['<', '3.5'])
    @expectedFailureAll(compiler="gcc", compiler_version=[
            '>=', '4.8.2'], archs=["i386"])
    @expectedFailureAll(compiler="gcc", compiler_version=[
            '<', '4.9'], archs=["x86_64"])
    def test_and_run_command(self):
        """Test expressions on register values."""

        # This test now ensures that each probable
        # register variable location is actually a register, and
        # if so, whether we can print out the variable there.
        # It only requires one of them to be handled in a non-error
        # way.
        register_variables_count = 0

        self.build()
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break inside the main.
        lldbutil.run_break_set_by_source_regexp(
            self, "break", num_expected_locations=3)

        ####################
        # First breakpoint

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        self.expect("breakpoint list -f", BREAKPOINT_HIT_ONCE,
                    substrs=[' resolved, hit count = 1'])

        # Try some variables that should be visible
        frame = self.dbg.GetSelectedTarget().GetProcess(
        ).GetSelectedThread().GetSelectedFrame()
        if self.is_variable_in_register(frame, 'a'):
            register_variables_count += 1
            self.expect("expr a", VARIABLES_DISPLAYED_CORRECTLY,
                        patterns=[re_expr_equals('int', 2)])

        if self.is_struct_pointer_in_register(frame, 'b', self.TraceOn()):
            register_variables_count += 1
            self.expect("expr b->m1", VARIABLES_DISPLAYED_CORRECTLY,
                        patterns=[re_expr_equals('int', 3)])

        #####################
        # Second breakpoint

        self.runCmd("continue")

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        self.expect("breakpoint list -f", BREAKPOINT_HIT_ONCE,
                    substrs=[' resolved, hit count = 1'])

        # Try some variables that should be visible
        frame = self.dbg.GetSelectedTarget().GetProcess(
        ).GetSelectedThread().GetSelectedFrame()
        if self.is_struct_pointer_in_register(frame, 'b', self.TraceOn()):
            register_variables_count += 1
            self.expect("expr b->m2", VARIABLES_DISPLAYED_CORRECTLY,
                        patterns=[re_expr_equals('int', 5)])

        if self.is_variable_in_register(frame, 'c'):
            register_variables_count += 1
            self.expect("expr c", VARIABLES_DISPLAYED_CORRECTLY,
                        patterns=[re_expr_equals('int', 5)])

        #####################
        # Third breakpoint

        self.runCmd("continue")

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        self.expect("breakpoint list -f", BREAKPOINT_HIT_ONCE,
                    substrs=[' resolved, hit count = 1'])

        # Try some variables that should be visible
        frame = self.dbg.GetSelectedTarget().GetProcess(
        ).GetSelectedThread().GetSelectedFrame()
        if self.is_variable_in_register(frame, 'f'):
            register_variables_count += 1
            self.expect("expr f", VARIABLES_DISPLAYED_CORRECTLY,
                        patterns=[re_expr_equals('float', '3.1')])

        # Validate that we verified at least one register variable
        self.assertTrue(
            register_variables_count > 0,
            "expected to verify at least one variable in a register")
        self.trace("executed {} expressions with values in registers".format(register_variables_count))

        self.runCmd("kill")


    def is_variable_in_register(self, frame, var_name):
        # Ensure we can lookup the variable.
        var = frame.FindVariable(var_name)
        self.trace("\nchecking {}...".format(var_name))
        if var is None or not var.IsValid():
            self.trace("{} cannot be found".format(var_name))
            return False

        # Check that we can get its value.  If not, this
        # may be a variable that is just out of scope at this point.
        value = var.GetValue()
        self.trace("checking value...")
        if value is None:
            self.trace("value is invalid")
            return False
        else:
            self.trace("value is {}".format(value))

        # We have a variable and we can get its value.  The variable is in a
        # register if we cannot get an address for it, assuming it is not a
        # struct pointer.  (This is an approximation - compilers can do other
        # things with spitting up a value into multiple parts of multiple
        # registers, but what we're verifying here is much more than it was
        # doing before).
        var_addr = var.GetAddress()
        self.trace("checking address...")
        if var_addr.IsValid():
            # We have an address, it must not be in a register.
            self.trace("var {} is not in a register: has a valid address {}".format(var_name, var_addr))
            return False
        else:
            # We don't have an address but we can read the value.
            # It is likely stored in a register.
            self.trace("var {} is in a register (we don't have an address for it)".format(var_name))
            return True


    def is_struct_pointer_in_register(self, frame, var_name, trace):
        # Ensure we can lookup the variable.
        var = frame.FindVariable(var_name)
        if trace:
            print("\nchecking {}...".format(var_name))

        if var is None or not var.IsValid():
            self.trace("{} cannot be found".format(var_name))
            return False

        # Check that we can get its value.  If not, this
        # may be a variable that is just out of scope at this point.
        value = var.GetValue()
        self.trace("checking value...")
        if value is None:
            if trace:
                print("value is invalid")
            return False
        else:
            if trace:
                print("value is {}".format(value))

        var_loc = var.GetLocation()
        if trace:
            print("checking location: {}".format(var_loc))
        if var_loc is None or var_loc.startswith("0x"):
            # The frame var is not in a register but rather a memory location.
            self.trace("frame var {} is not in a register".format(var_name))
            return False
        else:
            self.trace("frame var {} is in a register".format(var_name))
            return True

