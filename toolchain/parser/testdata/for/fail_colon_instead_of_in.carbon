// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{carbon-run-parser}
// CHECK:STDOUT: [
// CHECK:STDOUT: {node_index: 20, kind: 'FunctionDefinition', text: '}', subtree_size: 21, children: [
// CHECK:STDOUT:   {node_index: 4, kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5, children: [
// CHECK:STDOUT:     {node_index: 0, kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:     {node_index: 1, kind: 'DeclaredName', text: 'foo'},
// CHECK:STDOUT:     {node_index: 3, kind: 'ParameterList', text: '(', subtree_size: 2, children: [
// CHECK:STDOUT:       {node_index: 2, kind: 'ParameterListEnd', text: ')'}]}]},
// CHECK:STDOUT:   {node_index: 19, kind: 'ForStatement', text: 'for', subtree_size: 15, children: [
// CHECK:STDOUT:     {node_index: 11, kind: 'ForHeader', text: '(', has_error: yes, subtree_size: 7, children: [
// CHECK:STDOUT:       {node_index: 8, kind: 'VariableDeclaration', text: 'var', subtree_size: 4, children: [
// CHECK:STDOUT:         {node_index: 7, kind: 'PatternBinding', text: ':', subtree_size: 3, children: [
// CHECK:STDOUT:           {node_index: 5, kind: 'DeclaredName', text: 'x'},
// CHECK:STDOUT:           {node_index: 6, kind: 'Literal', text: 'i32'}]}]},
// CHECK:STDOUT:       {node_index: 9, kind: 'NameReference', text: 'y'},
// CHECK:STDOUT:       {node_index: 10, kind: 'ForHeaderEnd', text: ')'}]},
// CHECK:STDOUT:     {node_index: 18, kind: 'CodeBlock', text: '}', subtree_size: 7, children: [
// CHECK:STDOUT:       {node_index: 12, kind: 'CodeBlockStart', text: '{'},
// CHECK:STDOUT:       {node_index: 17, kind: 'ExpressionStatement', text: ';', subtree_size: 5, children: [
// CHECK:STDOUT:         {node_index: 16, kind: 'CallExpression', text: '(', subtree_size: 4, children: [
// CHECK:STDOUT:           {node_index: 13, kind: 'NameReference', text: 'Print'},
// CHECK:STDOUT:           {node_index: 14, kind: 'NameReference', text: 'x'},
// CHECK:STDOUT:           {node_index: 15, kind: 'CallExpressionEnd', text: ')'}]}]}]}]}]},
// CHECK:STDOUT: {node_index: 21, kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

fn foo() {
  // CHECK:STDERR: {{.*}}/toolchain/parser/testdata/for/fail_colon_instead_of_in.carbon:[[@LINE+1]]:19: `:` should be replaced by `in`.
  for (var x: i32 : y) {
    Print(x);
  }
}
