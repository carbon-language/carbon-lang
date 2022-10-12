// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{carbon} dump parse-tree %s 2>&1 | %{FileCheck-strict} %s
// AUTOUPDATE: %{carbon} dump parse-tree %s

// CHECK:{{.*}}/toolchain/parser/testdata/function/declaration/with_identifier_as_param.carbon:[[@LINE+9]]:8: Expected parameter declaration.
// CHECK:[
// CHECK:{node_index: 4, kind: 'FunctionDeclaration', text: 'fn', subtree_size: 5, children: [
// CHECK:  {node_index: 0, kind: 'DeclaredName', text: 'foo'},
// CHECK:  {node_index: 2, kind: 'ParameterList', text: '(', has_error: yes, subtree_size: 2, children: [
// CHECK:    {node_index: 1, kind: 'ParameterListEnd', text: ')'}]},
// CHECK:  {node_index: 3, kind: 'DeclarationEnd', text: ';'}]},
// CHECK:{node_index: 5, kind: 'FileEnd', text: ''},
// CHECK:]
fn foo(bar);
