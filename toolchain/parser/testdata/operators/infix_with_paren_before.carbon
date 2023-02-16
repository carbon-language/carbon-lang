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
// CHECK:STDOUT:       {kind: 'ParenExpressionOrTupleLiteralStart', text: '('},
// CHECK:STDOUT:       {kind: 'NameReference', text: 'n'},
// CHECK:STDOUT:     {kind: 'ParenExpression', text: ')', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'Literal', text: '3'},
// CHECK:STDOUT:   {kind: 'InfixOperator', text: '*', subtree_size: 5},
// CHECK:STDOUT: {kind: 'VariableDeclaration', text: ';', subtree_size: 11},
// CHECK:STDOUT: {kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

var n: i8 = (n)*3;
