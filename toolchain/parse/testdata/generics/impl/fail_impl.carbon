// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_impl.carbon:[[@LINE+3]]:10: ERROR: Expected `as` in `impl` declaration.
// CHECK:STDERR: impl foo bar;
// CHECK:STDERR:          ^~~
impl foo bar;

impl

// CHECK:STDERR: fail_impl.carbon:[[@LINE+13]]:21: ERROR: Expected expression.
// CHECK:STDERR: // CHECK:STDOUT:   ]
// CHECK:STDERR:                     ^
// CHECK:STDOUT: - filename: fail_impl.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'ImplIntroducer', text: 'impl'},
// CHECK:STDOUT:       {kind: 'IdentifierNameExpr', text: 'foo'},
// CHECK:STDOUT:     {kind: 'ImplDecl', text: ';', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ImplIntroducer', text: 'impl'},
// CHECK:STDOUT:       {kind: 'InvalidParse', text: '', has_error: yes},
// CHECK:STDOUT:     {kind: 'ImplDecl', text: 'impl', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
