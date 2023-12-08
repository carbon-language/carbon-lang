// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_paren_match_regression.carbon:[[@LINE+6]]:5: ERROR: Expected binding pattern.
// CHECK:STDERR: var = (foo {})
// CHECK:STDERR:     ^
// CHECK:STDERR: fail_paren_match_regression.carbon:[[@LINE+3]]:12: ERROR: Expected `,` or `)`.
// CHECK:STDERR: var = (foo {})
// CHECK:STDERR:            ^
var = (foo {})

// CHECK:STDERR: fail_paren_match_regression.carbon:[[@LINE+16]]:21: ERROR: `var` declarations must end with a `;`.
// CHECK:STDERR: // CHECK:STDOUT:   ]
// CHECK:STDERR:                     ^
// CHECK:STDOUT: - filename: fail_paren_match_regression.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: '=', has_error: yes},
// CHECK:STDOUT:         {kind: 'InvalidParse', text: '=', has_error: yes},
// CHECK:STDOUT:       {kind: 'BindingPattern', text: '=', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:       {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'ExprOpenParen', text: '('},
// CHECK:STDOUT:         {kind: 'IdentifierNameExpr', text: 'foo'},
// CHECK:STDOUT:       {kind: 'ParenExpr', text: ')', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: 'var', has_error: yes, subtree_size: 9},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
