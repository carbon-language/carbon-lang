// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{carbon} dump parse-tree %s | %{FileCheck-strict} %s
// AUTOUPDATE: %{carbon} dump parse-tree %s
// CHECK:[
// CHECK:{node_index: 13, kind: 'FunctionDeclaration', text: 'fn', subtree_size: 14, children: [
// CHECK:  {node_index: 0, kind: 'DeclaredName', text: 'F'},
// CHECK:  {node_index: 2, kind: 'ParameterList', text: '(', subtree_size: 2, children: [
// CHECK:    {node_index: 1, kind: 'ParameterListEnd', text: ')'}]},
// CHECK:  {node_index: 12, kind: 'CodeBlock', text: '{', subtree_size: 10, children: [
// CHECK:    {node_index: 10, kind: 'IfStatement', text: 'if', subtree_size: 8, children: [
// CHECK:      {node_index: 5, kind: 'Condition', text: '(', subtree_size: 3, children: [
// CHECK:        {node_index: 3, kind: 'NameReference', text: 'c'},
// CHECK:        {node_index: 4, kind: 'ConditionEnd', text: ')'}]},
// CHECK:      {node_index: 9, kind: 'CodeBlock', text: '{', subtree_size: 4, children: [
// CHECK:        {node_index: 7, kind: 'ReturnStatement', text: 'return', subtree_size: 2, children: [
// CHECK:          {node_index: 6, kind: 'StatementEnd', text: ';'}]},
// CHECK:        {node_index: 8, kind: 'CodeBlockEnd', text: '}'}]}]},
// CHECK:    {node_index: 11, kind: 'CodeBlockEnd', text: '}'}]}]},
// CHECK:{node_index: 27, kind: 'FunctionDeclaration', text: 'fn', subtree_size: 14, children: [
// CHECK:  {node_index: 14, kind: 'DeclaredName', text: 'G'},
// CHECK:  {node_index: 19, kind: 'ParameterList', text: '(', subtree_size: 5, children: [
// CHECK:    {node_index: 17, kind: 'PatternBinding', text: ':', subtree_size: 3, children: [
// CHECK:      {node_index: 15, kind: 'DeclaredName', text: 'x'},
// CHECK:      {node_index: 16, kind: 'NameReference', text: 'Foo'}]},
// CHECK:    {node_index: 18, kind: 'ParameterListEnd', text: ')'}]},
// CHECK:  {node_index: 21, kind: 'ReturnType', text: '->', subtree_size: 2, children: [
// CHECK:    {node_index: 20, kind: 'NameReference', text: 'Foo'}]},
// CHECK:  {node_index: 26, kind: 'CodeBlock', text: '{', subtree_size: 5, children: [
// CHECK:    {node_index: 24, kind: 'ReturnStatement', text: 'return', subtree_size: 3, children: [
// CHECK:      {node_index: 22, kind: 'NameReference', text: 'x'},
// CHECK:      {node_index: 23, kind: 'StatementEnd', text: ';'}]},
// CHECK:    {node_index: 25, kind: 'CodeBlockEnd', text: '}'}]}]},
// CHECK:{node_index: 28, kind: 'FileEnd', text: ''},
// CHECK:]

// NOTE: Move to its own directory when more tests are added.
fn F() {
  if (c) {
    return;
  }
}
fn G(x: Foo) -> Foo {
  return x;
}
