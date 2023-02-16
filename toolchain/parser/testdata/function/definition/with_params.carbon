// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{carbon-run-parser}
// CHECK:STDOUT: [
// CHECK:STDOUT:     {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:     {kind: 'DeclaredName', text: 'foo'},
// CHECK:STDOUT:       {kind: 'ParameterListStart', text: '('},
// CHECK:STDOUT:         {kind: 'DeclaredName', text: 'bar'},
// CHECK:STDOUT:         {kind: 'Literal', text: 'i64'},
// CHECK:STDOUT:       {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ParameterListComma', text: ','},
// CHECK:STDOUT:         {kind: 'DeclaredName', text: 'baz'},
// CHECK:STDOUT:         {kind: 'Literal', text: 'i64'},
// CHECK:STDOUT:       {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'ParameterList', text: ')', subtree_size: 9},
// CHECK:STDOUT:   {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 12},
// CHECK:STDOUT:         {kind: 'NameReference', text: 'foo'},
// CHECK:STDOUT:       {kind: 'CallExpressionStart', text: '(', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'NameReference', text: 'baz'},
// CHECK:STDOUT:       {kind: 'CallExpressionComma', text: ','},
// CHECK:STDOUT:         {kind: 'NameReference', text: 'bar'},
// CHECK:STDOUT:         {kind: 'NameReference', text: 'baz'},
// CHECK:STDOUT:       {kind: 'InfixOperator', text: '+', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'CallExpression', text: ')', subtree_size: 8},
// CHECK:STDOUT:   {kind: 'ExpressionStatement', text: ';', subtree_size: 9},
// CHECK:STDOUT: {kind: 'FunctionDefinition', text: '}', subtree_size: 22},
// CHECK:STDOUT: {kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

fn foo(bar: i64, baz: i64) {
  foo(baz, bar + baz);
}
