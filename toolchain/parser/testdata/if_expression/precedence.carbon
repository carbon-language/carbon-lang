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
// CHECK:STDOUT:         {kind: 'Literal', text: 'bool'},
// CHECK:STDOUT:       {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'ParameterList', text: ')', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'Literal', text: 'bool'},
// CHECK:STDOUT:     {kind: 'ReturnType', text: '->', subtree_size: 2},
// CHECK:STDOUT:   {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 10},
// CHECK:STDOUT:     {kind: 'ReturnStatementStart', text: 'return'},
// CHECK:STDOUT:           {kind: 'NameReference', text: 'b'},
// CHECK:STDOUT:           {kind: 'NameReference', text: 'b'},
// CHECK:STDOUT:         {kind: 'InfixOperator', text: 'and', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'IfExpressionIf', text: 'if', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'NameReference', text: 'b'},
// CHECK:STDOUT:           {kind: 'NameReference', text: 'b'},
// CHECK:STDOUT:         {kind: 'InfixOperator', text: 'and', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'IfExpressionThen', text: 'then', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'NameReference', text: 'b'},
// CHECK:STDOUT:         {kind: 'NameReference', text: 'b'},
// CHECK:STDOUT:       {kind: 'InfixOperator', text: 'or', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'IfExpressionElse', text: 'else', subtree_size: 12},
// CHECK:STDOUT:   {kind: 'ReturnStatement', text: ';', subtree_size: 14},
// CHECK:STDOUT: {kind: 'FunctionDefinition', text: '}', subtree_size: 25},
// CHECK:STDOUT: {kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

fn F(b: bool) -> bool {
  return if b and b then b and b else b or b;
}
