// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{carbon} dump parse-tree %s | %{FileCheck-strict} %s
// AUTOUPDATE: %{carbon} dump parse-tree %s

// CHECK:{{.*}}/toolchain/parser/testdata/function/declaration/skip_without_semi_to_curly.carbon:[[@LINE+9]]:1: Unrecognized declaration introducer.
// CHECK:[
// CHECK:{node_index: 4, kind: 'FunctionDeclaration', text: 'fn', subtree_size: 5, children: [
// CHECK:  {node_index: 0, kind: 'DeclaredName', text: 'F'},
// CHECK:  {node_index: 2, kind: 'ParameterList', text: '(', subtree_size: 2, children: [
// CHECK:    {node_index: 1, kind: 'ParameterListEnd', text: ')'}]},
// CHECK:  {node_index: 3, kind: 'DeclarationEnd', text: ';'}]},
// CHECK:{node_index: 5, kind: 'FileEnd', text: ''},
// CHECK:]
struct X { fn () }
fn F();
