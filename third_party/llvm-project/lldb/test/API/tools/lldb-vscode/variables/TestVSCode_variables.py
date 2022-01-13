"""
Test lldb-vscode setBreakpoints request
"""

from __future__ import print_function

import unittest2
import vscode
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import lldbvscode_testcase


def make_buffer_verify_dict(start_idx, count, offset=0):
    verify_dict = {}
    for i in range(start_idx, start_idx + count):
        verify_dict['[%i]' % (i)] = {'type': 'int', 'value': str(i+offset)}
    return verify_dict


class TestVSCode_variables(lldbvscode_testcase.VSCodeTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)

    def verify_values(self, verify_dict, actual, varref_dict=None, expression=None):
        if 'equals' in verify_dict:
            verify = verify_dict['equals']
            for key in verify:
                verify_value = verify[key]
                actual_value = actual[key]
                self.assertEqual(verify_value, actual_value,
                                '"%s" keys don\'t match (%s != %s)' % (
                                    key, actual_value, verify_value))
        if 'startswith' in verify_dict:
            verify = verify_dict['startswith']
            for key in verify:
                verify_value = verify[key]
                actual_value = actual[key]
                startswith = actual_value.startswith(verify_value)
                self.assertTrue(startswith,
                                ('"%s" value "%s" doesn\'t start with'
                                 ' "%s")') % (
                                    key, actual_value,
                                    verify_value))
        hasVariablesReference = 'variablesReference' in actual
        varRef = None
        if hasVariablesReference:
            # Remember variable references in case we want to test further
            # by using the evaluate name.
            varRef = actual["variablesReference"]
            if varRef != 0 and varref_dict is not None:
                if expression is None:
                    evaluateName = actual["evaluateName"]
                else:
                    evaluateName = expression
                varref_dict[evaluateName] = varRef
        if ('hasVariablesReference' in verify_dict and
                verify_dict['hasVariablesReference']):
            self.assertTrue(hasVariablesReference,
                            "verify variable reference")
        if 'children' in verify_dict:
            self.assertTrue(hasVariablesReference and varRef is not None and
                            varRef != 0,
                            ("children verify values specified for "
                             "variable without children"))

            response = self.vscode.request_variables(varRef)
            self.verify_variables(verify_dict['children'],
                                  response['body']['variables'],
                                  varref_dict)

    def verify_variables(self, verify_dict, variables, varref_dict=None):
        for variable in variables:
            name = variable['name']
            self.assertIn(name, verify_dict,
                          'variable "%s" in verify dictionary' % (name))
            self.verify_values(verify_dict[name], variable, varref_dict)

    @skipIfWindows
    @skipIfRemote
    def test_scopes_variables_setVariable_evaluate(self):
        '''
            Tests the "scopes", "variables", "setVariable", and "evaluate"
            packets.
        '''
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)
        source = 'main.cpp'
        breakpoint1_line = line_number(source, '// breakpoint 1')
        lines = [breakpoint1_line]
        # Set breakpoint in the thread function so we can step the threads
        breakpoint_ids = self.set_source_breakpoints(source, lines)
        self.assertEqual(len(breakpoint_ids), len(lines),
                        "expect correct number of breakpoints")
        self.continue_to_breakpoints(breakpoint_ids)
        locals = self.vscode.get_local_variables()
        globals = self.vscode.get_global_variables()
        buffer_children = make_buffer_verify_dict(0, 32)
        verify_locals = {
            'argc': {
                'equals': {'type': 'int', 'value': '1'}
            },
            'argv': {
                'equals': {'type': 'const char **'},
                'startswith': {'value': '0x'},
                'hasVariablesReference': True
            },
            'pt': {
                'equals': {'type': 'PointType'},
                'hasVariablesReference': True,
                'children': {
                    'x': {'equals': {'type': 'int', 'value': '11'}},
                    'y': {'equals': {'type': 'int', 'value': '22'}},
                    'buffer': {'children': buffer_children}
                }
            },
            'x': {
                'equals': {'type': 'int'}
            }
        }
        verify_globals = {
            's_local': {
                'equals': {'type': 'float', 'value': '2.25'}
            },
            '::g_global': {
                'equals': {'type': 'int', 'value': '123'}
            },
            's_global': {
                'equals': {'type': 'int', 'value': '234'}
            },
        }
        varref_dict = {}
        self.verify_variables(verify_locals, locals, varref_dict)
        self.verify_variables(verify_globals, globals, varref_dict)
        # pprint.PrettyPrinter(indent=4).pprint(varref_dict)
        # We need to test the functionality of the "variables" request as it
        # has optional parameters like "start" and "count" to limit the number
        # of variables that are fetched
        varRef = varref_dict['pt.buffer']
        response = self.vscode.request_variables(varRef)
        self.verify_variables(buffer_children, response['body']['variables'])
        # Verify setting start=0 in the arguments still gets all children
        response = self.vscode.request_variables(varRef, start=0)
        self.verify_variables(buffer_children, response['body']['variables'])
        # Verify setting count=0 in the arguments still gets all children.
        # If count is zero, it means to get all children.
        response = self.vscode.request_variables(varRef, count=0)
        self.verify_variables(buffer_children, response['body']['variables'])
        # Verify setting count to a value that is too large in the arguments
        # still gets all children, and no more
        response = self.vscode.request_variables(varRef, count=1000)
        self.verify_variables(buffer_children, response['body']['variables'])
        # Verify setting the start index and count gets only the children we
        # want
        response = self.vscode.request_variables(varRef, start=5, count=5)
        self.verify_variables(make_buffer_verify_dict(5, 5),
                              response['body']['variables'])
        # Verify setting the start index to a value that is out of range
        # results in an empty list
        response = self.vscode.request_variables(varRef, start=32, count=1)
        self.assertEqual(len(response['body']['variables']), 0,
                        'verify we get no variable back for invalid start')

        # Test evaluate
        expressions = {
            'pt.x': {
                'equals': {'result': '11', 'type': 'int'},
                'hasVariablesReference': False
            },
            'pt.buffer[2]': {
                'equals': {'result': '2', 'type': 'int'},
                'hasVariablesReference': False
            },
            'pt': {
                'equals': {'type': 'PointType'},
                'startswith': {'result': 'PointType @ 0x'},
                'hasVariablesReference': True
            },
            'pt.buffer': {
                'equals': {'type': 'int[32]'},
                'startswith': {'result': 'int[32] @ 0x'},
                'hasVariablesReference': True
            },
            'argv': {
                'equals': {'type': 'const char **'},
                'startswith': {'result': '0x'},
                'hasVariablesReference': True
            },
            'argv[0]': {
                'equals': {'type': 'const char *'},
                'startswith': {'result': '0x'},
                'hasVariablesReference': True
            },
            '2+3': {
                'equals': {'result': '5', 'type': 'int'},
                'hasVariablesReference': False
            },
        }
        for expression in expressions:
            response = self.vscode.request_evaluate(expression)
            self.verify_values(expressions[expression], response['body'])

        # Test setting variables
        self.set_local('argc', 123)
        argc = self.get_local_as_int('argc')
        self.assertEqual(argc, 123,
                        'verify argc was set to 123 (123 != %i)' % (argc))

        self.set_local('argv', 0x1234)
        argv = self.get_local_as_int('argv')
        self.assertEqual(argv, 0x1234,
                        'verify argv was set to 0x1234 (0x1234 != %#x)' % (
                            argv))

        # Set a variable value whose name is synthetic, like a variable index
        # and verify the value by reading it
        self.vscode.request_setVariable(varRef, "[0]", 100)
        response = self.vscode.request_variables(varRef, start=0, count=1)
        self.verify_variables(make_buffer_verify_dict(0, 1, 100),
                              response['body']['variables'])

        # Set a variable value whose name is a real child value, like "pt.x"
        # and verify the value by reading it
        varRef = varref_dict['pt']
        self.vscode.request_setVariable(varRef, "x", 111)
        response = self.vscode.request_variables(varRef, start=0, count=1)
        value = response['body']['variables'][0]['value']
        self.assertEqual(value, '111',
                        'verify pt.x got set to 111 (111 != %s)' % (value))

        # We check shadowed variables and that a new get_local_variables request
        # gets the right data
        breakpoint2_line = line_number(source, '// breakpoint 2')
        lines = [breakpoint2_line]
        breakpoint_ids = self.set_source_breakpoints(source, lines)
        self.assertEqual(len(breakpoint_ids), len(lines),
                        "expect correct number of breakpoints")
        self.continue_to_breakpoints(breakpoint_ids)

        verify_locals['argc']['equals']['value'] = '123'
        verify_locals['pt']['children']['x']['equals']['value'] = '111'
        verify_locals['x @ main.cpp:17'] = {'equals': {'type': 'int', 'value': '89'}}
        verify_locals['x @ main.cpp:19'] = {'equals': {'type': 'int', 'value': '42'}}
        verify_locals['x @ main.cpp:21'] = {'equals': {'type': 'int', 'value': '72'}}

        self.verify_variables(verify_locals, self.vscode.get_local_variables())

        # Now we verify that we correctly change the name of a variable with and without differentiator suffix
        self.assertFalse(self.vscode.request_setVariable(1, "x2", 9)['success'])
        self.assertFalse(self.vscode.request_setVariable(1, "x @ main.cpp:0", 9)['success'])

        self.assertTrue(self.vscode.request_setVariable(1, "x @ main.cpp:17", 17)['success'])
        self.assertTrue(self.vscode.request_setVariable(1, "x @ main.cpp:19", 19)['success'])
        self.assertTrue(self.vscode.request_setVariable(1, "x @ main.cpp:21", 21)['success'])

        # The following should have no effect
        self.assertFalse(self.vscode.request_setVariable(1, "x @ main.cpp:21", "invalid")['success'])

        verify_locals['x @ main.cpp:17']['equals']['value'] = '17'
        verify_locals['x @ main.cpp:19']['equals']['value'] = '19'
        verify_locals['x @ main.cpp:21']['equals']['value'] = '21'

        self.verify_variables(verify_locals, self.vscode.get_local_variables())

        # The plain x variable shold refer to the innermost x
        self.assertTrue(self.vscode.request_setVariable(1, "x", 22)['success'])
        verify_locals['x @ main.cpp:21']['equals']['value'] = '22'

        self.verify_variables(verify_locals, self.vscode.get_local_variables())

        # In breakpoint 3, there should be no shadowed variables
        breakpoint3_line = line_number(source, '// breakpoint 3')
        lines = [breakpoint3_line]
        breakpoint_ids = self.set_source_breakpoints(source, lines)
        self.assertEqual(len(breakpoint_ids), len(lines),
                        "expect correct number of breakpoints")
        self.continue_to_breakpoints(breakpoint_ids)

        locals = self.vscode.get_local_variables()
        names = [var['name'] for var in locals]
        # The first shadowed x shouldn't have a suffix anymore
        verify_locals['x'] = {'equals': {'type': 'int', 'value': '17'}}
        self.assertNotIn('x @ main.cpp:17', names)
        self.assertNotIn('x @ main.cpp:19', names)
        self.assertNotIn('x @ main.cpp:21', names)

        self.verify_variables(verify_locals, locals)

    @skipIfWindows
    @skipIfRemote
    def test_scopes_and_evaluate_expansion(self):
        """
        Tests the evaluated expression expands successfully after "scopes" packets
        and permanent expressions persist.
        """
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)
        source = "main.cpp"
        breakpoint1_line = line_number(source, "// breakpoint 1")
        lines = [breakpoint1_line]
        # Set breakpoint in the thread function so we can step the threads
        breakpoint_ids = self.set_source_breakpoints(source, lines)
        self.assertEqual(
            len(breakpoint_ids), len(lines), "expect correct number of breakpoints"
        )
        self.continue_to_breakpoints(breakpoint_ids)

        # Verify locals
        locals = self.vscode.get_local_variables()
        buffer_children = make_buffer_verify_dict(0, 32)
        verify_locals = {
            "argc": {"equals": {"type": "int", "value": "1"}},
            "argv": {
                "equals": {"type": "const char **"},
                "startswith": {"value": "0x"},
                "hasVariablesReference": True,
            },
            "pt": {
                "equals": {"type": "PointType"},
                "hasVariablesReference": True,
                "children": {
                    "x": {"equals": {"type": "int", "value": "11"}},
                    "y": {"equals": {"type": "int", "value": "22"}},
                    "buffer": {"children": buffer_children},
                },
            },
            "x": {"equals": {"type": "int"}},
        }
        self.verify_variables(verify_locals, locals)

        # Evaluate expandable expression twice: once permanent (from repl)
        # the other temporary (from other UI).
        expandable_expression = {
            "name": "pt",
            "response": {
                "equals": {"type": "PointType"},
                "startswith": {"result": "PointType @ 0x"},
                "hasVariablesReference": True,
            },
            "children": {
                "x": {"equals": {"type": "int", "value": "11"}},
                "y": {"equals": {"type": "int", "value": "22"}},
                "buffer": {"children": buffer_children},
            },
        }

        # Evaluate from permanent UI.
        permanent_expr_varref_dict = {}
        response = self.vscode.request_evaluate(
            expandable_expression["name"], frameIndex=0, threadId=None, context="repl"
        )
        self.verify_values(
            expandable_expression["response"],
            response["body"],
            permanent_expr_varref_dict,
            expandable_expression["name"],
        )

        # Evaluate from temporary UI.
        temporary_expr_varref_dict = {}
        response = self.vscode.request_evaluate(expandable_expression["name"])
        self.verify_values(
            expandable_expression["response"],
            response["body"],
            temporary_expr_varref_dict,
            expandable_expression["name"],
        )

        # Evaluate locals again.
        locals = self.vscode.get_local_variables()
        self.verify_variables(verify_locals, locals)

        # Verify the evaluated expressions before second locals evaluation
        # can be expanded.
        var_ref = temporary_expr_varref_dict[expandable_expression["name"]]
        response = self.vscode.request_variables(var_ref)
        self.verify_variables(
            expandable_expression["children"], response["body"]["variables"]
        )

        # Continue to breakpoint 3, permanent variable should still exist
        # after resume.
        breakpoint3_line = line_number(source, "// breakpoint 3")
        lines = [breakpoint3_line]
        breakpoint_ids = self.set_source_breakpoints(source, lines)
        self.assertEqual(
            len(breakpoint_ids), len(lines), "expect correct number of breakpoints"
        )
        self.continue_to_breakpoints(breakpoint_ids)

        var_ref = permanent_expr_varref_dict[expandable_expression["name"]]
        response = self.vscode.request_variables(var_ref)
        self.verify_variables(
            expandable_expression["children"], response["body"]["variables"]
        )

        # Test that frame scopes have corresponding presentation hints.
        frame_id = self.vscode.get_stackFrame()["id"]
        scopes = self.vscode.request_scopes(frame_id)["body"]["scopes"]

        scope_names = [scope["name"] for scope in scopes]
        self.assertIn("Locals", scope_names)
        self.assertIn("Registers", scope_names)

        for scope in scopes:
            if scope["name"] == "Locals":
                self.assertEquals(scope.get("presentationHint"), "locals")
            if scope["name"] == "Registers":
                self.assertEquals(scope.get("presentationHint"), "registers")
