// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{carbon} dump parse-tree %s | %{FileCheck-strict} %s
// AUTOUPDATE: %{carbon} dump parse-tree %s
// CHECK:[
// CHECK:{node_index: 6, kind: 'VariableDeclaration', text: 'var', subtree_size: 7, children: [
// CHECK:  {node_index: 2, kind: 'PatternBinding', text: ':', subtree_size: 3, children: [
// CHECK:    {node_index: 0, kind: 'DeclaredName', text: 'v'},
// CHECK:    {node_index: 1, kind: 'Literal', text: 'i32'}]},
// CHECK:  {node_index: 4, kind: 'VariableInitializer', text: '=', subtree_size: 2, children: [
// CHECK:    {node_index: 3, kind: 'Literal', text: '0'}]},
// CHECK:  {node_index: 5, kind: 'DeclarationEnd', text: ';'}]},
// CHECK:{node_index: 11, kind: 'VariableDeclaration', text: 'var', subtree_size: 5, children: [
// CHECK:  {node_index: 9, kind: 'PatternBinding', text: ':', subtree_size: 3, children: [
// CHECK:    {node_index: 7, kind: 'DeclaredName', text: 'w'},
// CHECK:    {node_index: 8, kind: 'Literal', text: 'i32'}]},
// CHECK:  {node_index: 10, kind: 'DeclarationEnd', text: ';'}]},
// CHECK:{node_index: 24, kind: 'FunctionDeclaration', text: 'fn', subtree_size: 13, children: [
// CHECK:  {node_index: 12, kind: 'DeclaredName', text: 'F'},
// CHECK:  {node_index: 14, kind: 'ParameterList', text: '(', subtree_size: 2, children: [
// CHECK:    {node_index: 13, kind: 'ParameterListEnd', text: ')'}]},
// CHECK:  {node_index: 23, kind: 'CodeBlock', text: '{', subtree_size: 9, children: [
// CHECK:    {node_index: 21, kind: 'VariableDeclaration', text: 'var', subtree_size: 7, children: [
// CHECK:      {node_index: 17, kind: 'PatternBinding', text: ':', subtree_size: 3, children: [
// CHECK:        {node_index: 15, kind: 'DeclaredName', text: 's'},
// CHECK:        {node_index: 16, kind: 'NameReference', text: 'String'}]},
// CHECK:      {node_index: 19, kind: 'VariableInitializer', text: '=', subtree_size: 2, children: [
// CHECK:        {node_index: 18, kind: 'Literal', text: '"hello"'}]},
// CHECK:      {node_index: 20, kind: 'DeclarationEnd', text: ';'}]},
// CHECK:    {node_index: 22, kind: 'CodeBlockEnd', text: '}'}]}]},
// CHECK:{node_index: 25, kind: 'FileEnd', text: ''},
// CHECK:]

// NOTE: Move to its own directory when more tests are added.
var v: i32 = 0;
var w: i32;
fn F() {
  var s: String = "hello";
}
