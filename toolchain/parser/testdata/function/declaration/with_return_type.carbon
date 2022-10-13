// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{carbon} dump parse-tree %s | %{FileCheck-strict} %s
// AUTOUPDATE: %{carbon} dump parse-tree %s
// CHECK:[
// CHECK:{node_index: 6, kind: 'FunctionDeclaration', text: 'fn', subtree_size: 7, children: [
// CHECK:  {node_index: 0, kind: 'DeclaredName', text: 'foo'},
// CHECK:  {node_index: 2, kind: 'ParameterList', text: '(', subtree_size: 2, children: [
// CHECK:    {node_index: 1, kind: 'ParameterListEnd', text: ')'}]},
// CHECK:  {node_index: 4, kind: 'ReturnType', text: '->', subtree_size: 2, children: [
// CHECK:    {node_index: 3, kind: 'Literal', text: 'u32'}]},
// CHECK:  {node_index: 5, kind: 'DeclarationEnd', text: ';'}]},
// CHECK:{node_index: 7, kind: 'FileEnd', text: ''},
// CHECK:]

fn foo() -> u32;
