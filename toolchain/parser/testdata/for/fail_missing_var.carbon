// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{carbon-run-parser}
// CHECK:STDOUT: [
// CHECK:STDOUT: {node_index: 17, kind: 'FunctionDefinition', text: '}', subtree_size: 18, children: [
// CHECK:STDOUT:   {node_index: 4, kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5, children: [
// CHECK:STDOUT:     {node_index: 0, kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:     {node_index: 1, kind: 'DeclaredName', text: 'foo'},
// CHECK:STDOUT:     {node_index: 3, kind: 'ParameterList', text: '(', subtree_size: 2, children: [
// CHECK:STDOUT:       {node_index: 2, kind: 'ParameterListEnd', text: ')'}]}]},
// CHECK:STDOUT:   {node_index: 16, kind: 'ForStatement', text: 'for', subtree_size: 12, children: [
// CHECK:STDOUT:     {node_index: 8, kind: 'ForHeader', text: '(', has_error: yes, subtree_size: 4, children: [
// CHECK:STDOUT:       {node_index: 5, kind: 'ForIn', text: 'in'},
// CHECK:STDOUT:       {node_index: 6, kind: 'NameReference', text: 'y'},
// CHECK:STDOUT:       {node_index: 7, kind: 'ForHeaderEnd', text: ')'}]},
// CHECK:STDOUT:     {node_index: 15, kind: 'CodeBlock', text: '}', subtree_size: 7, children: [
// CHECK:STDOUT:       {node_index: 9, kind: 'CodeBlockStart', text: '{'},
// CHECK:STDOUT:       {node_index: 14, kind: 'ExpressionStatement', text: ';', subtree_size: 5, children: [
// CHECK:STDOUT:         {node_index: 13, kind: 'CallExpression', text: '(', subtree_size: 4, children: [
// CHECK:STDOUT:           {node_index: 10, kind: 'NameReference', text: 'Print'},
// CHECK:STDOUT:           {node_index: 11, kind: 'NameReference', text: 'x'},
// CHECK:STDOUT:           {node_index: 12, kind: 'CallExpressionEnd', text: ')'}]}]}]}]}]},
// CHECK:STDOUT: {node_index: 18, kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

fn foo() {
  // CHECK:STDERR: {{.*}}/toolchain/parser/testdata/for/fail_missing_var.carbon:[[@LINE+1]]:8: Expected `var` declaration.
  for (x: i32 in y) {
    Print(x);
  }
}
