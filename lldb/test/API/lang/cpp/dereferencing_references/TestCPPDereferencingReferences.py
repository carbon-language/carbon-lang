import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def test(self):
        """Tests deferencing lvalue/rvalue references via LLDB's builtin type system."""
        self.build()
        lldbutil.run_to_source_breakpoint(self, "// break here", lldb.SBFileSpec("main.cpp"))

        # Take an lvalue reference and call `Dereference` on the SBValue.
        # The result should be `TTT` (and *not* for example the underlying type
        # 'int').
        lref_val = self.expect_var_path("l_ref", type="TTT &")
        self.assertEqual(lref_val.Dereference().GetType().GetName(), "TTT")

        # Same as above for rvalue references.
        rref_val = self.expect_var_path("r_ref", type="TTT &&")
        self.assertEqual(rref_val.Dereference().GetType().GetName(), "TTT")
