// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{carbon-run-parser}
// CHECK:STDOUT: [
// CHECK:STDOUT: {node_index: 6, kind: 'FunctionDeclaration', text: ';', subtree_size: 7, children: [
// CHECK:STDOUT:   {node_index: 0, kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:   {node_index: 1, kind: 'DeclaredName', text: 'foo'},
// CHECK:STDOUT:   {node_index: 3, kind: 'ParameterList', text: '(', subtree_size: 2, children: [
// CHECK:STDOUT:     {node_index: 2, kind: 'ParameterListEnd', text: ')'}]},
// CHECK:STDOUT:   {node_index: 5, kind: 'ReturnType', text: '->', subtree_size: 2, children: [
// CHECK:STDOUT:     {node_index: 4, kind: 'Literal', text: 'u32'}]}]},
// CHECK:STDOUT: {node_index: 7, kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

fn foo() -> u32;
