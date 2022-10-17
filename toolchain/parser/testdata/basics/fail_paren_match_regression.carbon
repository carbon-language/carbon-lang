// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{carbon-run-parser}
// CHECK:STDOUT: [
// CHECK:STDOUT: {node_index: 4, kind: 'VariableDeclaration', text: 'var', has_error: yes, subtree_size: 5, children: [
// CHECK:STDOUT:   {node_index: 3, kind: 'VariableInitializer', text: '=', subtree_size: 4, children: [
// CHECK:STDOUT:     {node_index: 2, kind: 'ParenExpression', text: '(', has_error: yes, subtree_size: 3, children: [
// CHECK:STDOUT:       {node_index: 0, kind: 'NameReference', text: 'foo'},
// CHECK:STDOUT:       {node_index: 1, kind: 'ParenExpressionEnd', text: ')'}]}]}]},
// CHECK:STDOUT: {node_index: 5, kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

// CHECK:STDERR: {{.*}}/toolchain/parser/testdata/basics/fail_paren_match_regression.carbon:[[@LINE+3]]:5: Expected pattern in `var` declaration.
// CHECK:STDERR: {{.*}}/toolchain/parser/testdata/basics/fail_paren_match_regression.carbon:[[@LINE+2]]:12: Expected `,` or `)`.
// CHECK:STDERR: {{.*}}/toolchain/parser/testdata/basics/fail_paren_match_regression.carbon:[[@LINE+1]]:15: Expected `;` after expression.
var = (foo {})
