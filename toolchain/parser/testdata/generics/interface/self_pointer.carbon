// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: [
// CHECK:STDOUT:     {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:     {kind: 'DeclaredName', text: 'Foo'},
// CHECK:STDOUT:   {kind: 'InterfaceDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:     {kind: 'DeclaredName', text: 'Sub'},
// CHECK:STDOUT:       {kind: 'DeducedParameterListStart', text: '['},
// CHECK:STDOUT:           {kind: 'SelfValueIdentifier', text: 'self'},
// CHECK:STDOUT:             {kind: 'SelfTypeIdentifier', text: 'Self'},
// CHECK:STDOUT:           {kind: 'PostfixOperator', text: '*', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'PatternBinding', text: ':', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'Address', text: 'addr', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'DeducedParameterList', text: ']', subtree_size: 7},
// CHECK:STDOUT:       {kind: 'ParameterListStart', text: '('},
// CHECK:STDOUT:         {kind: 'DeclaredName', text: 'b'},
// CHECK:STDOUT:         {kind: 'SelfTypeIdentifier', text: 'Self'},
// CHECK:STDOUT:       {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'ParameterList', text: ')', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'SelfTypeIdentifier', text: 'Self'},
// CHECK:STDOUT:     {kind: 'ReturnType', text: '->', subtree_size: 2},
// CHECK:STDOUT:   {kind: 'FunctionDeclaration', text: ';', subtree_size: 17},
// CHECK:STDOUT:     {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:     {kind: 'DeclaredName', text: 'Sub'},
// CHECK:STDOUT:       {kind: 'DeducedParameterListStart', text: '['},
// CHECK:STDOUT:           {kind: 'SelfValueIdentifier', text: 'self'},
// CHECK:STDOUT:             {kind: 'NameReference', text: 'foo'},
// CHECK:STDOUT:           {kind: 'PostfixOperator', text: '*', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'PatternBinding', text: ':', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'Address', text: 'addr', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'DeducedParameterList', text: ']', subtree_size: 7},
// CHECK:STDOUT:       {kind: 'ParameterListStart', text: '('},
// CHECK:STDOUT:         {kind: 'DeclaredName', text: 'b'},
// CHECK:STDOUT:         {kind: 'SelfTypeIdentifier', text: 'Self'},
// CHECK:STDOUT:       {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'ParameterList', text: ')', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'SelfTypeIdentifier', text: 'Self'},
// CHECK:STDOUT:     {kind: 'ReturnType', text: '->', subtree_size: 2},
// CHECK:STDOUT:   {kind: 'FunctionDeclaration', text: ';', subtree_size: 17},
// CHECK:STDOUT: {kind: 'InterfaceDefinition', text: '}', subtree_size: 38},
// CHECK:STDOUT: {kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

interface Foo {
  fn Sub[addr self: Self*](b: Self) -> Self;
  fn Sub[addr self: foo*](b: Self) -> Self;
}
