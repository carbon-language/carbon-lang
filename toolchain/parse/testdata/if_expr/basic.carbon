// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F(b: bool, x: i32, y: i32) -> bool {
  return if b then x else y;
}

// CHECK:STDOUT: - filename: basic.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'F'},
// CHECK:STDOUT:           {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:             {kind: 'IdentifierName', text: 'b'},
// CHECK:STDOUT:             {kind: 'BoolTypeLiteral', text: 'bool'},
// CHECK:STDOUT:           {kind: 'BindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:             {kind: 'IdentifierName', text: 'x'},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'BindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:             {kind: 'IdentifierName', text: 'y'},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'BindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'TuplePattern', text: ')', subtree_size: 13},
// CHECK:STDOUT:           {kind: 'BoolTypeLiteral', text: 'bool'},
// CHECK:STDOUT:         {kind: 'ReturnType', text: '->', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 18},
// CHECK:STDOUT:         {kind: 'ReturnStatementStart', text: 'return'},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'b'},
// CHECK:STDOUT:           {kind: 'IfExprIf', text: 'if', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'x'},
// CHECK:STDOUT:           {kind: 'IfExprThen', text: 'then', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'y'},
// CHECK:STDOUT:         {kind: 'IfExprElse', text: 'else', subtree_size: 6},
// CHECK:STDOUT:       {kind: 'ReturnStatement', text: ';', subtree_size: 8},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 27},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
