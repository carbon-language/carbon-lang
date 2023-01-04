// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{carbon-run-parser}
// CHECK:STDOUT: [
// CHECK:STDOUT:   {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:     {kind: 'DeclaredName', text: 'n'},
// CHECK:STDOUT:     {kind: 'Literal', text: 'i8'},
// CHECK:STDOUT:   {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:   {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:       {kind: 'NameReference', text: 'F'},
// CHECK:STDOUT:     {kind: 'CallExpressionStart', text: '(', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'Literal', text: 'i8'},
// CHECK:STDOUT:     {kind: 'PostfixOperator', text: '*', subtree_size: 2},
// CHECK:STDOUT:   {kind: 'CallExpression', text: ')', subtree_size: 5},
// CHECK:STDOUT: {kind: 'VariableDeclaration', text: ';', subtree_size: 11},
// CHECK:STDOUT: {kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

// CHECK:STDERR: {{.*}}/toolchain/parser/testdata/operators/recover_postfix_space_in_call.carbon:[[@LINE+1]]:18: Whitespace is not allowed before this unary operator.
var n: i8 = F(i8 *);
