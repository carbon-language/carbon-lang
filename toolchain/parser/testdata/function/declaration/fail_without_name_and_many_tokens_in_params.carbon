// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{carbon-run-parser}
// CHECK:STDOUT: [
// CHECK:STDOUT: {node_index: 1, kind: 'FunctionDeclaration', text: 'fn', has_error: yes, subtree_size: 2, children: [
// CHECK:STDOUT:   {node_index: 0, kind: 'DeclarationEnd', text: ';'}]},
// CHECK:STDOUT: {node_index: 2, kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

// CHECK:STDERR: {{.*}}/toolchain/parser/testdata/function/declaration/fail_without_name_and_many_tokens_in_params.carbon:[[@LINE+1]]:4: Expected function name after `fn` keyword.
fn (a tokens c d e f g h i j k l m n o p q r s t u v w x y z);
