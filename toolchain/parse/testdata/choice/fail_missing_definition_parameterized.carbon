// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_missing_definition_parameterized.carbon:[[@LINE+3]]:35: ERROR: Choice definition expected.
// CHECK:STDERR: choice MissingDefinition(T:! type);
// CHECK:STDERR:                                   ^
choice MissingDefinition(T:! type);

// CHECK:STDOUT: - filename: fail_missing_definition_parameterized.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'ChoiceIntroducer', text: 'choice'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'MissingDefinition'},
// CHECK:STDOUT:           {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:             {kind: 'IdentifierName', text: 'T'},
// CHECK:STDOUT:             {kind: 'TypeTypeLiteral', text: 'type'},
// CHECK:STDOUT:           {kind: 'GenericBindingPattern', text: ':!', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'TuplePattern', text: ')', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ChoiceDefinitionStart', text: ';', has_error: yes, subtree_size: 8},
// CHECK:STDOUT:     {kind: 'ChoiceDefinition', text: ';', has_error: yes, subtree_size: 9},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
