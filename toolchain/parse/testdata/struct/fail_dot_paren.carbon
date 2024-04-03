// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_dot_paren.carbon:[[@LINE+3]]:10: ERROR: Expected identifier after `.`.
// CHECK:STDERR: var x: {.(a) = 1};
// CHECK:STDERR:          ^
var x: {.(a) = 1};

// CHECK:STDOUT: - filename: fail_dot_paren.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'x'},
// CHECK:STDOUT:           {kind: 'StructLiteralStart', text: '{'},
// CHECK:STDOUT:             {kind: 'IdentifierName', text: '(', has_error: yes},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '1'},
// CHECK:STDOUT:           {kind: 'InvalidParse', text: '=', has_error: yes},
// CHECK:STDOUT:         {kind: 'StructLiteral', text: '}', has_error: yes, subtree_size: 6},
// CHECK:STDOUT:       {kind: 'BindingPattern', text: ':', subtree_size: 8},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: ';', subtree_size: 10},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
