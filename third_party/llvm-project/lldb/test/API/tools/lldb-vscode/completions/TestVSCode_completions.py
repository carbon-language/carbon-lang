"""
Test lldb-vscode completions request
"""


import lldbvscode_testcase
import unittest2
import vscode
from lldbsuite.test import lldbutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class TestVSCode_variables(lldbvscode_testcase.VSCodeTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)

    def verify_completions(self, actual_list, expected_list, not_expected_list=[]):
        for expected_item in expected_list:
            self.assertIn(expected_item, actual_list)

        for not_expected_item in not_expected_list:
            self.assertNotIn(not_expected_item, actual_list)

    @skipIfWindows
    @skipIfDarwin # Skip this test for now until we can figure out why tings aren't working on build bots
    def test_completions(self):
        """
            Tests the completion request at different breakpoints
        """
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)
        source = "main.cpp"
        breakpoint1_line = line_number(source, "// breakpoint 1")
        breakpoint2_line = line_number(source, "// breakpoint 2")
        breakpoint_ids = self.set_source_breakpoints(
            source, [breakpoint1_line, breakpoint2_line]
        )
        self.continue_to_next_stop()

        # shouldn't see variables inside main
        self.verify_completions(
            self.vscode.get_completions("var"),
            [
                {
                    "text": "var",
                    "label": "var -- vector<basic_string<char, char_traits<char>, allocator<char>>, allocator<basic_string<char, char_traits<char>, allocator<char>>>> &",
                }
            ],
            [{"text": "var1", "label": "var1 -- int &"}],
        )

        # should see global keywords but not variables inside main
        self.verify_completions(
            self.vscode.get_completions("str"),
            [{"text": "struct", "label": "struct"}],
            [{"text": "str1", "label": "str1 -- std::string &"}],
        )

        self.continue_to_next_stop()

        # should see variables from main but not from the other function
        self.verify_completions(
            self.vscode.get_completions("var"),
            [
                {"text": "var1", "label": "var1 -- int &"},
                {"text": "var2", "label": "var2 -- int &"},
            ],
            [
                {
                    "text": "var",
                    "label": "var -- vector<basic_string<char, char_traits<char>, allocator<char> >, allocator<basic_string<char, char_traits<char>, allocator<char> > > > &",
                }
            ],
        )

        self.verify_completions(
            self.vscode.get_completions("str"),
            [
                {"text": "struct", "label": "struct"},
                {"text": "str1", "label": "str1 -- string &"},
            ],
        )

        # should complete arbitrary commands including word starts
        self.verify_completions(
            self.vscode.get_completions("`log enable  "),
            [{"text": "gdb-remote", "label": "gdb-remote"}],
        )

        # should complete expressions with quotes inside
        self.verify_completions(
            self.vscode.get_completions('`expr " "; typed'),
            [{"text": "typedef", "label": "typedef"}],
        )

        # should complete an incomplete quoted token
        self.verify_completions(
            self.vscode.get_completions('`setting "se'),
            [
                {
                    "text": "set",
                    "label": "set -- Set the value of the specified debugger setting.",
                }
            ],
        )
        self.verify_completions(
            self.vscode.get_completions("`'comm"),
            [
                {
                    "text": "command",
                    "label": "command -- Commands for managing custom LLDB commands.",
                }
            ],
        )

        self.verify_completions(
            self.vscode.get_completions("foo1.v"),
            [
                {
                    "text": "var1",
                    "label": "foo1.var1 -- int"
                }
            ]
        )

        self.verify_completions(
            self.vscode.get_completions("foo1.my_bar_object.v"),
            [
                {
                    "text": "var1",
                    "label": "foo1.my_bar_object.var1 -- int"
                }
            ]
        )

        self.verify_completions(
            self.vscode.get_completions("foo1.var1 + foo1.v"),
            [
                {
                    "text": "var1",
                    "label": "foo1.var1 -- int"
                }
            ]
        )

        self.verify_completions(
            self.vscode.get_completions("foo1.var1 + v"),
            [
                {
                    "text": "var1",
                    "label": "var1 -- int &"
                }
            ]
        )

        #should correctly handle spaces between objects and member operators
        self.verify_completions(
            self.vscode.get_completions("foo1 .v"),
            [
                {
                    "text": "var1",
                    "label": ".var1 -- int"
                }
            ],
            [
                {
                    "text": "var2",
                    "label": ".var2 -- int"
                }
            ]
        )

        self.verify_completions(
            self.vscode.get_completions("foo1 . v"),
            [
                {
                    "text": "var1",
                    "label": "var1 -- int"
                }
            ], 
            [
                {
                    "text": "var2",
                    "label": "var2 -- int"
                }
            ]
        )
