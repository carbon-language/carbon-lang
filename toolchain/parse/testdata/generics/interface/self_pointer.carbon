// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

interface Foo {
  fn Sub[addr self: Self*](b: Self) -> Self;
  fn Sub[addr self: foo*](b: Self) -> Self;
}

// CHECK:STDOUT: - filename: self_pointer.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'Foo'},
// CHECK:STDOUT:       {kind: 'InterfaceDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'Sub'},
// CHECK:STDOUT:           {kind: 'ImplicitParamListStart', text: '['},
// CHECK:STDOUT:               {kind: 'SelfValueName', text: 'self'},
// CHECK:STDOUT:                 {kind: 'SelfTypeNameExpr', text: 'Self'},
// CHECK:STDOUT:               {kind: 'PostfixOperator', text: '*', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'BindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'Address', text: 'addr', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'ImplicitParamList', text: ']', subtree_size: 7},
// CHECK:STDOUT:           {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:             {kind: 'IdentifierName', text: 'b'},
// CHECK:STDOUT:             {kind: 'SelfTypeNameExpr', text: 'Self'},
// CHECK:STDOUT:           {kind: 'BindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'TuplePattern', text: ')', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'SelfTypeNameExpr', text: 'Self'},
// CHECK:STDOUT:         {kind: 'ReturnType', text: '->', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDecl', text: ';', subtree_size: 17},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'Sub'},
// CHECK:STDOUT:           {kind: 'ImplicitParamListStart', text: '['},
// CHECK:STDOUT:               {kind: 'SelfValueName', text: 'self'},
// CHECK:STDOUT:                 {kind: 'IdentifierNameExpr', text: 'foo'},
// CHECK:STDOUT:               {kind: 'PostfixOperator', text: '*', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'BindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'Address', text: 'addr', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'ImplicitParamList', text: ']', subtree_size: 7},
// CHECK:STDOUT:           {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:             {kind: 'IdentifierName', text: 'b'},
// CHECK:STDOUT:             {kind: 'SelfTypeNameExpr', text: 'Self'},
// CHECK:STDOUT:           {kind: 'BindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'TuplePattern', text: ')', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'SelfTypeNameExpr', text: 'Self'},
// CHECK:STDOUT:         {kind: 'ReturnType', text: '->', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDecl', text: ';', subtree_size: 17},
// CHECK:STDOUT:     {kind: 'InterfaceDefinition', text: '}', subtree_size: 38},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
