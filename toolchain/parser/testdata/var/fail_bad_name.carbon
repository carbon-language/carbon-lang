// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: [
// CHECK:STDOUT:   {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:     {kind: 'DeclaredName', text: '*', has_error: yes},
// CHECK:STDOUT:     {kind: 'InvalidParse', text: '*', has_error: yes},
// CHECK:STDOUT:   {kind: 'PatternBinding', text: '*', has_error: yes, subtree_size: 3},
// CHECK:STDOUT: {kind: 'VariableDeclaration', text: ';', subtree_size: 5},
// CHECK:STDOUT: {kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

// CHECK:STDERR: toolchain/parser/testdata/var/fail_bad_name.carbon:[[@LINE+1]]:5: Expected pattern in `var` declaration.
var *;
