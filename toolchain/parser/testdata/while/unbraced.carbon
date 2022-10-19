// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// TODO: This should have an error.
// AUTOUPDATE
// RUN: %{carbon-run-parser}
// CHECK:STDOUT: [
// CHECK:STDOUT: {node_index: 11, kind: 'FunctionDeclaration', text: 'fn', subtree_size: 12, children: [
// CHECK:STDOUT:   {node_index: 0, kind: 'DeclaredName', text: 'F'},
// CHECK:STDOUT:   {node_index: 2, kind: 'ParameterList', text: '(', subtree_size: 2, children: [
// CHECK:STDOUT:     {node_index: 1, kind: 'ParameterListEnd', text: ')'}]},
// CHECK:STDOUT:   {node_index: 10, kind: 'CodeBlock', text: '{', subtree_size: 8, children: [
// CHECK:STDOUT:     {node_index: 8, kind: 'WhileStatement', text: 'while', subtree_size: 6, children: [
// CHECK:STDOUT:       {node_index: 5, kind: 'Condition', text: '(', subtree_size: 3, children: [
// CHECK:STDOUT:         {node_index: 3, kind: 'NameReference', text: 'a'},
// CHECK:STDOUT:         {node_index: 4, kind: 'ConditionEnd', text: ')'}]},
// CHECK:STDOUT:       {node_index: 7, kind: 'BreakStatement', text: 'break', subtree_size: 2, children: [
// CHECK:STDOUT:         {node_index: 6, kind: 'StatementEnd', text: ';'}]}]},
// CHECK:STDOUT:     {node_index: 9, kind: 'CodeBlockEnd', text: '}'}]}]},
// CHECK:STDOUT: {node_index: 12, kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

fn F() {
  while (a)
    // CHECK:STDERR: {{.*}}/toolchain/parser/testdata/while/unbraced.carbon:[[@LINE+1]]:5: Expected braced code block.
    break;
}
