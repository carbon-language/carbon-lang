"""
Test basic std::forward_list functionality.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestBasicForwardList(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @add_test_categories(["libc++"])
    @skipIf(compiler=no_match("clang"))
    def test(self):
        self.build()

        lldbutil.run_to_source_breakpoint(self,
                                          "// Set break point at this line.",
                                          lldb.SBFileSpec("main.cpp"))

        self.runCmd("settings set target.import-std-module true")

        list_type = "std::forward_list<int>"
        value_type = list_type + "::value_type"

        # FIXME: This has three elements in it but the formatter seems to
        # calculate the wrong size and contents.
        self.expect_expr("a", result_type=list_type, result_summary="size=1")
        self.expect_expr("std::distance(a.begin(), a.end())", result_value="3")
        self.expect_expr("a.front()", result_type=value_type, result_value="3")

        self.expect("expr a.sort()")
        self.expect_expr("a.front()", result_type=value_type, result_value="1")

        # FIXME: Removing casting here causes LLDB to crash.
        self.expect_expr("(int)*a.begin()",
                         result_type="int",
                         result_value="1")
