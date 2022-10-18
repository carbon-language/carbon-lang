// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{carbon-run-parser}
// CHECK:STDOUT: [
// CHECK:STDOUT: {node_index: 9, kind: 'FunctionDeclaration', text: 'fn', subtree_size: 10, children: [
// CHECK:STDOUT:   {node_index: 0, kind: 'DeclaredName', text: 'F'},
// CHECK:STDOUT:   {node_index: 2, kind: 'ParameterList', text: '(', subtree_size: 2, children: [
// CHECK:STDOUT:     {node_index: 1, kind: 'ParameterListEnd', text: ')'}]},
// CHECK:STDOUT:   {node_index: 8, kind: 'CodeBlock', text: '{', subtree_size: 6, children: [
// CHECK:STDOUT:     {node_index: 6, kind: 'ExpressionStatement', text: ';', subtree_size: 4, children: [
// CHECK:STDOUT:       {node_index: 5, kind: 'PostfixOperator', text: '++', subtree_size: 3, children: [
// CHECK:STDOUT:         {node_index: 4, kind: 'PostfixOperator', text: '++', subtree_size: 2, children: [
// CHECK:STDOUT:           {node_index: 3, kind: 'NameReference', text: 'n'}]}]}]},
// CHECK:STDOUT:     {node_index: 7, kind: 'CodeBlockEnd', text: '}'}]}]},
// CHECK:STDOUT: {node_index: 10, kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

fn F() {
  n++++;
}
