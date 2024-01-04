// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// Fix and uncomment this to test error handling.
// CHECK:STDERR: fail_missing_implicit_close.carbon:[[@LINE+6]]:7: ERROR: Opening symbol without a corresponding closing symbol.
// CHECK:STDERR: fn Div[();
// CHECK:STDERR:       ^
// CHECK:STDERR: fail_missing_implicit_close.carbon:[[@LINE+3]]:7: ERROR: `fn` requires a `(` for parameters.
// CHECK:STDERR: fn Div[();
// CHECK:STDERR:       ^
fn Div[();

// CHECK:STDOUT: - filename: fail_missing_implicit_close.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'IdentifierName', text: 'Div'},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
