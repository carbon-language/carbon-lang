// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: [
// CHECK:STDOUT:   {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:   {kind: 'DeclaredName', text: 'foo'},
// CHECK:STDOUT:     {kind: 'ParameterListStart', text: '('},
// CHECK:STDOUT:       {kind: 'DeclaredName', text: 'bar'},
// CHECK:STDOUT:       {kind: 'InvalidParse', text: ')', has_error: yes},
// CHECK:STDOUT:     {kind: 'PatternBinding', text: 'bar', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:   {kind: 'ParameterList', text: ')', has_error: yes, subtree_size: 5},
// CHECK:STDOUT: {kind: 'FunctionDeclaration', text: ';', subtree_size: 8},
// CHECK:STDOUT: {kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

// CHECK:STDERR: toolchain/parser/testdata/function/declaration/fail_with_identifier_as_param.carbon:[[@LINE+1]]:11: Expected parameter declaration.
fn foo(bar);
