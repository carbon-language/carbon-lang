// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{carbon} dump parse-tree %s | %{FileCheck-strict} %s
// AUTOUPDATE: %{carbon} dump parse-tree %s
// CHECK:[
// CHECK:{node_index: 21, kind: 'FunctionDeclaration', text: 'fn', subtree_size: 22, children: [
// CHECK:  {node_index: 0, kind: 'DeclaredName', text: 'foo'},
// CHECK:  {node_index: 9, kind: 'ParameterList', text: '(', subtree_size: 9, children: [
// CHECK:    {node_index: 3, kind: 'PatternBinding', text: ':', subtree_size: 3, children: [
// CHECK:      {node_index: 1, kind: 'DeclaredName', text: 'bar'},
// CHECK:      {node_index: 2, kind: 'Literal', text: 'i64'}]},
// CHECK:    {node_index: 4, kind: 'ParameterListComma', text: ','},
// CHECK:    {node_index: 7, kind: 'PatternBinding', text: ':', subtree_size: 3, children: [
// CHECK:      {node_index: 5, kind: 'DeclaredName', text: 'baz'},
// CHECK:      {node_index: 6, kind: 'Literal', text: 'i64'}]},
// CHECK:    {node_index: 8, kind: 'ParameterListEnd', text: ')'}]},
// CHECK:  {node_index: 20, kind: 'CodeBlock', text: '{', subtree_size: 11, children: [
// CHECK:    {node_index: 18, kind: 'ExpressionStatement', text: ';', subtree_size: 9, children: [
// CHECK:      {node_index: 17, kind: 'CallExpression', text: '(', subtree_size: 8, children: [
// CHECK:        {node_index: 10, kind: 'NameReference', text: 'foo'},
// CHECK:        {node_index: 11, kind: 'NameReference', text: 'baz'},
// CHECK:        {node_index: 12, kind: 'CallExpressionComma', text: ','},
// CHECK:        {node_index: 15, kind: 'InfixOperator', text: '+', subtree_size: 3, children: [
// CHECK:          {node_index: 13, kind: 'NameReference', text: 'bar'},
// CHECK:          {node_index: 14, kind: 'NameReference', text: 'baz'}]},
// CHECK:        {node_index: 16, kind: 'CallExpressionEnd', text: ')'}]}]},
// CHECK:    {node_index: 19, kind: 'CodeBlockEnd', text: '}'}]}]},
// CHECK:{node_index: 22, kind: 'FileEnd', text: ''},
// CHECK:]

fn foo(bar: i64, baz: i64) {
  foo(baz, bar + baz);
}
