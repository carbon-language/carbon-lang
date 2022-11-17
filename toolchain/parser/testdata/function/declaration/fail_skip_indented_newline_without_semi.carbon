// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{carbon-run-parser}
// CHECK:STDOUT: [
// CHECK:STDOUT:   {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT: {kind: 'FunctionDeclaration', text: 'fn', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:   {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:   {kind: 'DeclaredName', text: 'F'},
// CHECK:STDOUT:     {kind: 'ParameterListStart', text: '('},
// CHECK:STDOUT:   {kind: 'ParameterList', text: ')', subtree_size: 2},
// CHECK:STDOUT: {kind: 'FunctionDeclaration', text: ';', subtree_size: 5},
// CHECK:STDOUT: {kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

// CHECK:STDERR: {{.*}}/toolchain/parser/testdata/function/declaration/fail_skip_indented_newline_without_semi.carbon:[[@LINE+1]]:4: Expected function name after `fn` keyword.
fn (x,
    y,
    z)
fn F();
