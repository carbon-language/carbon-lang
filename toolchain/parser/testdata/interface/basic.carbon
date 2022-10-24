// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{carbon} dump parse-tree %s | %{FileCheck-strict} %s
// AUTOUPDATE: %{carbon} dump parse-tree %s
// CHECK:STDOUT: [
// CHECK:STDOUT: {node_index: 42, kind: 'InterfaceDeclaration', text: 'interface', subtree_size: 43, children: [
// CHECK:STDOUT:   {node_index: 0, kind: 'DeclaredName', text: 'foo'},
// CHECK:STDOUT:   {node_index: 41, kind: 'InterfaceBody', text: '{', subtree_size: 41, children: [
// CHECK:STDOUT:     {node_index: 15, kind: 'FunctionDeclaration', text: 'fn', subtree_size: 15, children: [
// CHECK:STDOUT:       {node_index: 1, kind: 'DeclaredName', text: 'Add'},
// CHECK:STDOUT:       {node_index: 6, kind: 'MeParam', text: '[', subtree_size: 5, children: [
// CHECK:STDOUT:         {node_index: 4, kind: 'PatternBinding', text: ':', subtree_size: 3, children: [
// CHECK:STDOUT:           {node_index: 2, kind: 'Me', text: 'me'},
// CHECK:STDOUT:           {node_index: 3, kind: 'Self', text: 'Self'}]},
// CHECK:STDOUT:         {node_index: 5, kind: 'MeParamEnd', text: ']'}]},
// CHECK:STDOUT:       {node_index: 11, kind: 'ParameterList', text: '(', subtree_size: 5, children: [
// CHECK:STDOUT:         {node_index: 9, kind: 'PatternBinding', text: ':', subtree_size: 3, children: [
// CHECK:STDOUT:           {node_index: 7, kind: 'DeclaredName', text: 'b'},
// CHECK:STDOUT:           {node_index: 8, kind: 'Self', text: 'Self'}]},
// CHECK:STDOUT:         {node_index: 10, kind: 'ParameterListEnd', text: ')'}]},
// CHECK:STDOUT:       {node_index: 13, kind: 'ReturnType', text: '->', subtree_size: 2, children: [
// CHECK:STDOUT:         {node_index: 12, kind: 'Self', text: 'Self'}]},
// CHECK:STDOUT:       {node_index: 14, kind: 'DeclarationEnd', text: ';'}]},
// CHECK:STDOUT:     {node_index: 32, kind: 'FunctionDeclaration', text: 'fn', subtree_size: 17, children: [
// CHECK:STDOUT:       {node_index: 16, kind: 'DeclaredName', text: 'Sub'},
// CHECK:STDOUT:       {node_index: 23, kind: 'MeParam', text: '[', subtree_size: 7, children: [
// CHECK:STDOUT:         {node_index: 21, kind: 'PatternBinding', text: ':', subtree_size: 5, children: [
// CHECK:STDOUT:           {node_index: 17, kind: 'Addr', text: 'addr'},
// CHECK:STDOUT:           {node_index: 18, kind: 'Me', text: 'me'},
// CHECK:STDOUT:           {node_index: 19, kind: 'Self', text: 'Self'},
// CHECK:STDOUT:           {node_index: 20, kind: 'SelfPointer', text: '*'}]},
// CHECK:STDOUT:         {node_index: 22, kind: 'MeParamEnd', text: ']'}]},
// CHECK:STDOUT:       {node_index: 28, kind: 'ParameterList', text: '(', subtree_size: 5, children: [
// CHECK:STDOUT:         {node_index: 26, kind: 'PatternBinding', text: ':', subtree_size: 3, children: [
// CHECK:STDOUT:           {node_index: 24, kind: 'DeclaredName', text: 'b'},
// CHECK:STDOUT:           {node_index: 25, kind: 'Self', text: 'Self'}]},
// CHECK:STDOUT:         {node_index: 27, kind: 'ParameterListEnd', text: ')'}]},
// CHECK:STDOUT:       {node_index: 30, kind: 'ReturnType', text: '->', subtree_size: 2, children: [
// CHECK:STDOUT:         {node_index: 29, kind: 'Self', text: 'Self'}]},
// CHECK:STDOUT:       {node_index: 31, kind: 'DeclarationEnd', text: ';'}]},
// CHECK:STDOUT:     {node_index: 39, kind: 'FunctionDeclaration', text: 'fn', subtree_size: 7, children: [
// CHECK:STDOUT:       {node_index: 33, kind: 'DeclaredName', text: 'FooFactory'},
// CHECK:STDOUT:       {node_index: 35, kind: 'ParameterList', text: '(', subtree_size: 2, children: [
// CHECK:STDOUT:         {node_index: 34, kind: 'ParameterListEnd', text: ')'}]},
// CHECK:STDOUT:       {node_index: 37, kind: 'ReturnType', text: '->', subtree_size: 2, children: [
// CHECK:STDOUT:         {node_index: 36, kind: 'Self', text: 'Self'}]},
// CHECK:STDOUT:       {node_index: 38, kind: 'DeclarationEnd', text: ';'}]},
// CHECK:STDOUT:     {node_index: 40, kind: 'InterfaceBodyEnd', text: '}'}]}]},
// CHECK:STDOUT: {node_index: 43, kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]
interface foo {
  fn Add[me: Self](b: Self) -> Self;
  fn Sub[addr me: Self*](b: Self) -> Self;
  fn FooFactory() -> Self;
}
