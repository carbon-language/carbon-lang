// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{carbon-run-parser}
// CHECK:STDOUT: [
// CHECK:STDOUT: {node_index: 21, kind: 'FunctionDefinition', text: '}', subtree_size: 22, children: [
// CHECK:STDOUT:   {node_index: 11, kind: 'FunctionDefinitionStart', text: '{', subtree_size: 12, children: [
// CHECK:STDOUT:     {node_index: 0, kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:     {node_index: 1, kind: 'DeclaredName', text: 'foo'},
// CHECK:STDOUT:     {node_index: 10, kind: 'ParameterList', text: '(', subtree_size: 9, children: [
// CHECK:STDOUT:       {node_index: 4, kind: 'PatternBinding', text: ':', subtree_size: 3, children: [
// CHECK:STDOUT:         {node_index: 2, kind: 'DeclaredName', text: 'bar'},
// CHECK:STDOUT:         {node_index: 3, kind: 'Literal', text: 'i64'}]},
// CHECK:STDOUT:       {node_index: 5, kind: 'ParameterListComma', text: ','},
// CHECK:STDOUT:       {node_index: 8, kind: 'PatternBinding', text: ':', subtree_size: 3, children: [
// CHECK:STDOUT:         {node_index: 6, kind: 'DeclaredName', text: 'baz'},
// CHECK:STDOUT:         {node_index: 7, kind: 'Literal', text: 'i64'}]},
// CHECK:STDOUT:       {node_index: 9, kind: 'ParameterListEnd', text: ')'}]}]},
// CHECK:STDOUT:   {node_index: 20, kind: 'ExpressionStatement', text: ';', subtree_size: 9, children: [
// CHECK:STDOUT:     {node_index: 19, kind: 'CallExpression', text: '(', subtree_size: 8, children: [
// CHECK:STDOUT:       {node_index: 12, kind: 'NameReference', text: 'foo'},
// CHECK:STDOUT:       {node_index: 13, kind: 'NameReference', text: 'baz'},
// CHECK:STDOUT:       {node_index: 14, kind: 'CallExpressionComma', text: ','},
// CHECK:STDOUT:       {node_index: 17, kind: 'InfixOperator', text: '+', subtree_size: 3, children: [
// CHECK:STDOUT:         {node_index: 15, kind: 'NameReference', text: 'bar'},
// CHECK:STDOUT:         {node_index: 16, kind: 'NameReference', text: 'baz'}]},
// CHECK:STDOUT:       {node_index: 18, kind: 'CallExpressionEnd', text: ')'}]}]}]},
// CHECK:STDOUT: {node_index: 22, kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

fn foo(bar: i64, baz: i64) {
  foo(baz, bar + baz);
}
