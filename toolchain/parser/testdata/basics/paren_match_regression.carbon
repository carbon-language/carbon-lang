// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{carbon} dump parse-tree %s 2>&1 | %{FileCheck-strict} %s
// AUTOUPDATE: %{carbon} dump parse-tree %s

// CHECK:{{.*}}/toolchain/parser/testdata/basics/paren_match_regression.carbon:[[@LINE+11]]:5: Expected pattern in `var` declaration.
// CHECK:{{.*}}/toolchain/parser/testdata/basics/paren_match_regression.carbon:[[@LINE+10]]:12: Expected `,` or `)`.
// CHECK:{{.*}}/toolchain/parser/testdata/basics/paren_match_regression.carbon:[[@LINE+9]]:15: Expected `;` after expression.
// CHECK:[
// CHECK:{node_index: 4, kind: 'VariableDeclaration', text: 'var', has_error: yes, subtree_size: 5, children: [
// CHECK:  {node_index: 3, kind: 'VariableInitializer', text: '=', subtree_size: 4, children: [
// CHECK:    {node_index: 2, kind: 'ParenExpression', text: '(', has_error: yes, subtree_size: 3, children: [
// CHECK:      {node_index: 0, kind: 'NameReference', text: 'foo'},
// CHECK:      {node_index: 1, kind: 'ParenExpressionEnd', text: ')'}]}]}]},
// CHECK:{node_index: 5, kind: 'FileEnd', text: ''},
// CHECK:]
var = (foo {})
