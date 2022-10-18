// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{carbon-run-parser}
// CHECK:STDOUT: [
// CHECK:STDOUT: {node_index: 13, kind: 'FunctionDefinition', text: '}', subtree_size: 14, children: [
// CHECK:STDOUT:   {node_index: 4, kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5, children: [
// CHECK:STDOUT:     {node_index: 0, kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:     {node_index: 1, kind: 'DeclaredName', text: 'F'},
// CHECK:STDOUT:     {node_index: 3, kind: 'ParameterList', text: '(', subtree_size: 2, children: [
// CHECK:STDOUT:       {node_index: 2, kind: 'ParameterListEnd', text: ')'}]}]},
// CHECK:STDOUT:   {node_index: 12, kind: 'IfStatement', text: 'if', subtree_size: 8, children: [
// CHECK:STDOUT:     {node_index: 7, kind: 'Condition', text: '(', subtree_size: 3, children: [
// CHECK:STDOUT:       {node_index: 5, kind: 'NameReference', text: 'c'},
// CHECK:STDOUT:       {node_index: 6, kind: 'ConditionEnd', text: ')'}]},
// CHECK:STDOUT:     {node_index: 11, kind: 'CodeBlock', text: '}', subtree_size: 4, children: [
// CHECK:STDOUT:       {node_index: 8, kind: 'CodeBlockStart', text: '{'},
// CHECK:STDOUT:       {node_index: 10, kind: 'ReturnStatement', text: 'return', subtree_size: 2, children: [
// CHECK:STDOUT:         {node_index: 9, kind: 'StatementEnd', text: ';'}]}]}]}]},
// CHECK:STDOUT: {node_index: 27, kind: 'FunctionDefinition', text: '}', subtree_size: 14, children: [
// CHECK:STDOUT:   {node_index: 23, kind: 'FunctionDefinitionStart', text: '{', subtree_size: 10, children: [
// CHECK:STDOUT:     {node_index: 14, kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:     {node_index: 15, kind: 'DeclaredName', text: 'G'},
// CHECK:STDOUT:     {node_index: 20, kind: 'ParameterList', text: '(', subtree_size: 5, children: [
// CHECK:STDOUT:       {node_index: 18, kind: 'PatternBinding', text: ':', subtree_size: 3, children: [
// CHECK:STDOUT:         {node_index: 16, kind: 'DeclaredName', text: 'x'},
// CHECK:STDOUT:         {node_index: 17, kind: 'NameReference', text: 'Foo'}]},
// CHECK:STDOUT:       {node_index: 19, kind: 'ParameterListEnd', text: ')'}]},
// CHECK:STDOUT:     {node_index: 22, kind: 'ReturnType', text: '->', subtree_size: 2, children: [
// CHECK:STDOUT:       {node_index: 21, kind: 'NameReference', text: 'Foo'}]}]},
// CHECK:STDOUT:   {node_index: 26, kind: 'ReturnStatement', text: 'return', subtree_size: 3, children: [
// CHECK:STDOUT:     {node_index: 24, kind: 'NameReference', text: 'x'},
// CHECK:STDOUT:     {node_index: 25, kind: 'StatementEnd', text: ';'}]}]},
// CHECK:STDOUT: {node_index: 28, kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

// NOTE: Move to its own directory when more tests are added.
fn F() {
  if (c) {
    return;
  }
}
fn G(x: Foo) -> Foo {
  return x;
}
