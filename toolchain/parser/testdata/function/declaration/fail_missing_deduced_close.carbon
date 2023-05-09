// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: [
// CHECK:STDOUT:   {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:   {kind: 'DeclaredName', text: 'Div'},
// CHECK:STDOUT:     {kind: 'DeducedParameterListStart', text: '['},
// CHECK:STDOUT:       {kind: 'DeclaredName', text: '(', has_error: yes},
// CHECK:STDOUT:       {kind: 'InvalidParse', text: '(', has_error: yes},
// CHECK:STDOUT:     {kind: 'PatternBinding', text: '(', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:   {kind: 'DeducedParameterList', text: ']', has_error: yes, subtree_size: 5},
// CHECK:STDOUT: {kind: 'FunctionDeclaration', text: 'fn', has_error: yes, subtree_size: 8},
// CHECK:STDOUT: {kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

// Fix and uncomment this to test error handling.
// CHECK:STDERR: toolchain/parser/testdata/function/declaration/fail_missing_deduced_close.carbon:[[@LINE+2]]:7: Closing symbol does not match most recent opening symbol.
// CHECK:STDERR: toolchain/parser/testdata/function/declaration/fail_missing_deduced_close.carbon:[[@LINE+1]]:8: Expected parameter declaration.
fn Div[();
// CHECK:STDERR: toolchain/parser/testdata/function/declaration/fail_missing_deduced_close.carbon:[[@LINE+0]]:174: A `(` for parameters is required after deduced parameters.
