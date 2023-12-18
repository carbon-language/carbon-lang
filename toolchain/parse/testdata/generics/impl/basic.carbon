// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

impl i32 as Interface {
  fn Add[self: Self](b: Self) -> Self;
}

// CHECK:STDOUT: - filename: basic.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'ImplIntroducer', text: 'impl'},
// CHECK:STDOUT:         {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'ImplAs', text: 'as'},
// CHECK:STDOUT:         {kind: 'IdentifierNameExpr', text: 'Interface'},
// CHECK:STDOUT:       {kind: 'ImplDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'Add'},
// CHECK:STDOUT:           {kind: 'ImplicitParamListStart', text: '['},
// CHECK:STDOUT:             {kind: 'SelfValueName', text: 'self'},
// CHECK:STDOUT:             {kind: 'SelfTypeNameExpr', text: 'Self'},
// CHECK:STDOUT:           {kind: 'BindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'ImplicitParamList', text: ']', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:             {kind: 'IdentifierName', text: 'b'},
// CHECK:STDOUT:             {kind: 'SelfTypeNameExpr', text: 'Self'},
// CHECK:STDOUT:           {kind: 'BindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'TuplePattern', text: ')', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'SelfTypeNameExpr', text: 'Self'},
// CHECK:STDOUT:         {kind: 'ReturnType', text: '->', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDecl', text: ';', subtree_size: 15},
// CHECK:STDOUT:     {kind: 'ImplDefinition', text: '}', subtree_size: 21},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
