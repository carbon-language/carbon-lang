// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{carbon-run-parser}
// CHECK:STDOUT: [
// CHECK:STDOUT: {node_index: 2, kind: 'FunctionDeclaration', text: ';', has_error: yes, subtree_size: 3, children: [
// CHECK:STDOUT:   {node_index: 0, kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:   {node_index: 1, kind: 'DeclaredName', text: 'foo'}]},
// CHECK:STDOUT: {node_index: 3, kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

// CHECK:STDERR: {{.*}}/toolchain/parser/testdata/function/declaration/fail_identifier_instead_of_sig.carbon:[[@LINE+1]]:8: Expected `(` after function name.
fn foo bar;
