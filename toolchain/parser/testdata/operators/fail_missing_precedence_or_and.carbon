// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{carbon-run-parser}
// CHECK:STDOUT: [
// CHECK:STDOUT: {node_index: 11, kind: 'FunctionDefinition', text: '}', subtree_size: 12, children: [
// CHECK:STDOUT:   {node_index: 4, kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5, children: [
// CHECK:STDOUT:     {node_index: 0, kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:     {node_index: 1, kind: 'DeclaredName', text: 'F'},
// CHECK:STDOUT:     {node_index: 3, kind: 'ParameterList', text: '(', subtree_size: 2, children: [
// CHECK:STDOUT:       {node_index: 2, kind: 'ParameterListEnd', text: ')'}]}]},
// CHECK:STDOUT:   {node_index: 10, kind: 'ExpressionStatement', text: ';', subtree_size: 6, children: [
// CHECK:STDOUT:     {node_index: 9, kind: 'InfixOperator', text: 'and', has_error: yes, subtree_size: 5, children: [
// CHECK:STDOUT:       {node_index: 7, kind: 'InfixOperator', text: 'or', subtree_size: 3, children: [
// CHECK:STDOUT:         {node_index: 5, kind: 'NameReference', text: 'a'},
// CHECK:STDOUT:         {node_index: 6, kind: 'NameReference', text: 'b'}]},
// CHECK:STDOUT:       {node_index: 8, kind: 'NameReference', text: 'c'}]}]}]},
// CHECK:STDOUT: {node_index: 12, kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

fn F() {
  // CHECK:STDERR: {{.*}}/toolchain/parser/testdata/operators/fail_missing_precedence_or_and.carbon:[[@LINE+1]]:10: Parentheses are required to disambiguate operator precedence.
  a or b and c;
}
