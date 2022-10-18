// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{carbon-run-parser}
// CHECK:STDOUT: [
// CHECK:STDOUT: {node_index: 1, kind: 'FunctionDeclaration', text: 'fn', has_error: yes, subtree_size: 2, children: [
// CHECK:STDOUT:   {node_index: 0, kind: 'FunctionIntroducer', text: 'fn'}]},
// CHECK:STDOUT: {node_index: 6, kind: 'FunctionDeclaration', text: ';', subtree_size: 5, children: [
// CHECK:STDOUT:   {node_index: 2, kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:   {node_index: 3, kind: 'DeclaredName', text: 'F'},
// CHECK:STDOUT:   {node_index: 5, kind: 'ParameterList', text: '(', subtree_size: 2, children: [
// CHECK:STDOUT:     {node_index: 4, kind: 'ParameterListEnd', text: ')'}]}]},
// CHECK:STDOUT: {node_index: 7, kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

// CHECK:STDERR: {{.*}}/toolchain/parser/testdata/function/declaration/fail_skip_to_newline_without_semi.carbon:[[@LINE+1]]:4: Expected function name after `fn` keyword.
fn ()
fn F();
