// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TODO: It should emit only one error message.

// CHECK:STDERR: fail_require_close_bracket.carbon:[[@LINE+6]]:8: ERROR: Opening symbol without a corresponding closing symbol.
// CHECK:STDERR: var x: [i32;;
// CHECK:STDERR:        ^
// CHECK:STDERR: fail_require_close_bracket.carbon:[[@LINE+3]]:8: ERROR: Expected expression.
// CHECK:STDERR: var x: [i32;;
// CHECK:STDERR:        ^
var x: [i32;;

// CHECK:STDOUT: - filename: fail_require_close_bracket.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'x'},
// CHECK:STDOUT:         {kind: 'InvalidParse', text: '[', has_error: yes},
// CHECK:STDOUT:       {kind: 'BindingPattern', text: ':', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: ';', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'EmptyDecl', text: ';'},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
