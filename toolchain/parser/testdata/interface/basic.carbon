// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{carbon} dump parse-tree %s | %{FileCheck-strict} %s
// AUTOUPDATE: %{carbon} dump parse-tree %s
// CHECK:STDOUT: [
// CHECK:STDOUT: {node_index: 25, kind: 'InterfaceDeclaration', text: 'interface', subtree_size: 26, children: [
// CHECK:STDOUT:   {node_index: 0, kind: 'DeclaredName', text: 'foo'},
// CHECK:STDOUT:   {node_index: 24, kind: 'InterfaceBody', text: '{', subtree_size: 24, children: [
// CHECK:STDOUT:     {node_index: 15, kind: 'FunctionDeclaration', text: 'fn', subtree_size: 15, children: [
// CHECK:STDOUT:       {node_index: 1, kind: 'DeclaredName', text: 'Add'},
// CHECK:STDOUT:       {node_index: 6, kind: 'MeParam', text: '[', subtree_size: 5, children: [
// CHECK:STDOUT:         {node_index: 2, kind: 'Me', text: 'me'},
// CHECK:STDOUT:         {node_index: 3, kind: 'MeColon', text: ':'},
// CHECK:STDOUT:         {node_index: 4, kind: 'Self', text: 'Self'},
// CHECK:STDOUT:         {node_index: 5, kind: 'MeParamEnd', text: ']'}]},
// CHECK:STDOUT:       {node_index: 11, kind: 'ParameterList', text: '(', subtree_size: 5, children: [
// CHECK:STDOUT:         {node_index: 9, kind: 'PatternBinding', text: ':', subtree_size: 3, children: [
// CHECK:STDOUT:           {node_index: 7, kind: 'DeclaredName', text: 'b'},
// CHECK:STDOUT:           {node_index: 8, kind: 'Self', text: 'Self'}]},
// CHECK:STDOUT:         {node_index: 10, kind: 'ParameterListEnd', text: ')'}]},
// CHECK:STDOUT:       {node_index: 13, kind: 'ReturnType', text: '->', subtree_size: 2, children: [
// CHECK:STDOUT:         {node_index: 12, kind: 'Self', text: 'Self'}]},
// CHECK:STDOUT:       {node_index: 14, kind: 'DeclarationEnd', text: ';'}]},
// CHECK:STDOUT:     {node_index: 22, kind: 'FunctionDeclaration', text: 'fn', subtree_size: 7, children: [
// CHECK:STDOUT:       {node_index: 16, kind: 'DeclaredName', text: 'FooFactory'},
// CHECK:STDOUT:       {node_index: 18, kind: 'ParameterList', text: '(', subtree_size: 2, children: [
// CHECK:STDOUT:         {node_index: 17, kind: 'ParameterListEnd', text: ')'}]},
// CHECK:STDOUT:       {node_index: 20, kind: 'ReturnType', text: '->', subtree_size: 2, children: [
// CHECK:STDOUT:         {node_index: 19, kind: 'Self', text: 'Self'}]},
// CHECK:STDOUT:       {node_index: 21, kind: 'DeclarationEnd', text: ';'}]},
// CHECK:STDOUT:     {node_index: 23, kind: 'InterfaceBodyEnd', text: '}'}]}]},
// CHECK:STDOUT: {node_index: 26, kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

interface foo {
  fn Add[me: Self](b: Self) -> Self;
  fn FooFactory() -> Self;
}
