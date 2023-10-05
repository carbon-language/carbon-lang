// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TODO: It should emit only one error message.

// CHECK:STDERR: fail_require_close_bracket.carbon:[[@LINE+9]]:8: ERROR: Closing symbol does not match most recent opening symbol.
// CHECK:STDERR: var x: [i32;;
// CHECK:STDERR:        ^
// CHECK:STDERR: fail_require_close_bracket.carbon:[[@LINE+6]]:13: ERROR: Expected expression.
// CHECK:STDERR: var x: [i32;;
// CHECK:STDERR:             ^
// CHECK:STDERR: fail_require_close_bracket.carbon:[[@LINE+3]]:13: ERROR: Unexpected tokens before `]`.
// CHECK:STDERR: var x: [i32;;
// CHECK:STDERR:             ^
var x: [i32;;

// CHECK:STDERR: fail_require_close_bracket.carbon:[[@LINE+16]]:21: ERROR: `var` declarations must end with a `;`.
// CHECK:STDERR: // CHECK:STDOUT:   ]
// CHECK:STDERR:                     ^
// CHECK:STDOUT: - filename: fail_require_close_bracket.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:         {kind: 'Name', text: 'x'},
// CHECK:STDOUT:             {kind: 'ArrayExpressionStart', text: '['},
// CHECK:STDOUT:             {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:           {kind: 'ArrayExpressionSemi', text: ';', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'InvalidParse', text: ';', has_error: yes},
// CHECK:STDOUT:         {kind: 'ArrayExpression', text: ']', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:       {kind: 'PatternBinding', text: ':', subtree_size: 7},
// CHECK:STDOUT:     {kind: 'VariableDeclaration', text: 'var', has_error: yes, subtree_size: 9},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
