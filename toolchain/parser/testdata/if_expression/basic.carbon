// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: [
// CHECK:STDOUT:     {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:     {kind: 'DeclaredName', text: 'F'},
// CHECK:STDOUT:       {kind: 'ParameterListStart', text: '('},
// CHECK:STDOUT:         {kind: 'DeclaredName', text: 'b'},
// CHECK:STDOUT:         {kind: 'NameReference', text: 'bool'},
// CHECK:STDOUT:       {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ParameterListComma', text: ','},
// CHECK:STDOUT:         {kind: 'DeclaredName', text: 'x'},
// CHECK:STDOUT:         {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:       {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ParameterListComma', text: ','},
// CHECK:STDOUT:         {kind: 'DeclaredName', text: 'y'},
// CHECK:STDOUT:         {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:       {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'ParameterList', text: ')', subtree_size: 13},
// CHECK:STDOUT:       {kind: 'NameReference', text: 'bool'},
// CHECK:STDOUT:     {kind: 'ReturnType', text: '->', subtree_size: 2},
// CHECK:STDOUT:   {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 18},
// CHECK:STDOUT:     {kind: 'ReturnStatementStart', text: 'return'},
// CHECK:STDOUT:       {kind: 'NameReference', text: 'b'},
// CHECK:STDOUT:       {kind: 'IfExpressionThen', text: 'then'},
// CHECK:STDOUT:       {kind: 'NameReference', text: 'x'},
// CHECK:STDOUT:       {kind: 'IfExpressionElse', text: 'else'},
// CHECK:STDOUT:       {kind: 'NameReference', text: 'y'},
// CHECK:STDOUT:     {kind: 'IfExpression', text: 'if', subtree_size: 6},
// CHECK:STDOUT:   {kind: 'ReturnStatement', text: ';', subtree_size: 8},
// CHECK:STDOUT: {kind: 'FunctionDefinition', text: '}', subtree_size: 27},
// CHECK:STDOUT: {kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

fn F(b: bool, x: i32, y: i32) -> bool {
  return if b then x else y;
}
