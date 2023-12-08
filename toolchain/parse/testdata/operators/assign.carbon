// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() {
  var a: i32 = 0;
  var b: i32 = 1;
  a = b;
  a *= b;
  a /= b;
  a += b;
  a -= b;
  a %= b;
  a &= b;
  a |= b;
  a ^= b;
  a <<= b;
  a >>= b;
  ++a;
  --a;
}

// CHECK:STDOUT: - filename: assign.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'F'},
// CHECK:STDOUT:           {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:         {kind: 'TuplePattern', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:           {kind: 'IdentifierName', text: 'a'},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'BindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'IntLiteral', text: '0'},
// CHECK:STDOUT:       {kind: 'VariableDecl', text: ';', subtree_size: 7},
// CHECK:STDOUT:         {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:           {kind: 'IdentifierName', text: 'b'},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'BindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'IntLiteral', text: '1'},
// CHECK:STDOUT:       {kind: 'VariableDecl', text: ';', subtree_size: 7},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'a'},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'b'},
// CHECK:STDOUT:         {kind: 'InfixOperator', text: '=', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'a'},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'b'},
// CHECK:STDOUT:         {kind: 'InfixOperator', text: '*=', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'a'},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'b'},
// CHECK:STDOUT:         {kind: 'InfixOperator', text: '/=', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'a'},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'b'},
// CHECK:STDOUT:         {kind: 'InfixOperator', text: '+=', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'a'},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'b'},
// CHECK:STDOUT:         {kind: 'InfixOperator', text: '-=', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'a'},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'b'},
// CHECK:STDOUT:         {kind: 'InfixOperator', text: '%=', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'a'},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'b'},
// CHECK:STDOUT:         {kind: 'InfixOperator', text: '&=', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'a'},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'b'},
// CHECK:STDOUT:         {kind: 'InfixOperator', text: '|=', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'a'},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'b'},
// CHECK:STDOUT:         {kind: 'InfixOperator', text: '^=', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'a'},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'b'},
// CHECK:STDOUT:         {kind: 'InfixOperator', text: '<<=', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'a'},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'b'},
// CHECK:STDOUT:         {kind: 'InfixOperator', text: '>>=', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'a'},
// CHECK:STDOUT:         {kind: 'PrefixOperator', text: '++', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'a'},
// CHECK:STDOUT:         {kind: 'PrefixOperator', text: '--', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 70},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
