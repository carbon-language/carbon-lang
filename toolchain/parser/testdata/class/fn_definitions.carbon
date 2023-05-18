// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: [
// CHECK:STDOUT:     {kind: 'ClassIntroducer', text: 'class'},
// CHECK:STDOUT:     {kind: 'DeclaredName', text: 'Foo'},
// CHECK:STDOUT:   {kind: 'ClassDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'DeclaredName', text: 'Make'},
// CHECK:STDOUT:         {kind: 'ParameterListStart', text: '('},
// CHECK:STDOUT:       {kind: 'ParameterList', text: ')', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'SelfTypeIdentifier', text: 'Self'},
// CHECK:STDOUT:       {kind: 'ReturnType', text: '->', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 7},
// CHECK:STDOUT:       {kind: 'ReturnStatementStart', text: 'return'},
// CHECK:STDOUT:         {kind: 'StructLiteralOrStructTypeLiteralStart', text: '{'},
// CHECK:STDOUT:             {kind: 'DesignatedName', text: 'x'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'Literal', text: '0'},
// CHECK:STDOUT:         {kind: 'StructFieldValue', text: '=', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'StructLiteral', text: '}', subtree_size: 6},
// CHECK:STDOUT:     {kind: 'ReturnStatement', text: ';', subtree_size: 8},
// CHECK:STDOUT:   {kind: 'FunctionDefinition', text: '}', subtree_size: 16},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'DeclaredName', text: 'Baz'},
// CHECK:STDOUT:         {kind: 'DeducedParameterListStart', text: '['},
// CHECK:STDOUT:           {kind: 'SelfValueIdentifier', text: 'self'},
// CHECK:STDOUT:           {kind: 'SelfTypeIdentifier', text: 'Self'},
// CHECK:STDOUT:         {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'DeducedParameterList', text: ']', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'ParameterListStart', text: '('},
// CHECK:STDOUT:       {kind: 'ParameterList', text: ')', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:       {kind: 'ReturnType', text: '->', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 12},
// CHECK:STDOUT:       {kind: 'ReturnStatementStart', text: 'return'},
// CHECK:STDOUT:         {kind: 'SelfValueIdentifier', text: 'self'},
// CHECK:STDOUT:         {kind: 'DesignatedName', text: 'x'},
// CHECK:STDOUT:       {kind: 'DesignatorExpression', text: '.', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'ReturnStatement', text: ';', subtree_size: 5},
// CHECK:STDOUT:   {kind: 'FunctionDefinition', text: '}', subtree_size: 18},
// CHECK:STDOUT: {kind: 'ClassDefinition', text: '}', subtree_size: 38},
// CHECK:STDOUT: {kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

class Foo {
  fn Make() -> Self { return {.x = 0}; }

  fn Baz[self: Self]() -> i32 { return self.x; }
}
