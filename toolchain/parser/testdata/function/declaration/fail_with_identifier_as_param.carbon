// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{carbon-run-parser}
// CHECK:STDOUT: [
// CHECK:STDOUT: {node_index: 4, kind: 'FunctionDeclaration', text: ';', subtree_size: 5, children: [
// CHECK:STDOUT:   {node_index: 0, kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:   {node_index: 1, kind: 'DeclaredName', text: 'foo'},
// CHECK:STDOUT:   {node_index: 3, kind: 'ParameterList', text: '(', has_error: yes, subtree_size: 2, children: [
// CHECK:STDOUT:     {node_index: 2, kind: 'ParameterListEnd', text: ')'}]}]},
// CHECK:STDOUT: {node_index: 5, kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

// CHECK:STDERR: {{.*}}/toolchain/parser/testdata/function/declaration/fail_with_identifier_as_param.carbon:[[@LINE+1]]:8: Expected parameter declaration.
fn foo(bar);
