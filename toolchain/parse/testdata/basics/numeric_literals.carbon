// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() {
  // 8 and 9 trigger special behavior in APInt when mishandling signed versus
  // unsigned, so we pay extra attention to those.
  var ints: [i32; 5] = (
    8,
    9,
    0x8,
    0b1000,
    39999999999999999993,
  );
  var floats: [f64; 7] = (
    0.9,
    8.0,
    80.0,
    1.0e7,
    1.0e8,
    1.0e-8,
    39999999999999999993.0e39999999999999999993,
  );
}

// CHECK:STDOUT: - filename: numeric_literals.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'Name', text: 'F'},
// CHECK:STDOUT:           {kind: 'ParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:           {kind: 'Name', text: 'ints'},
// CHECK:STDOUT:               {kind: 'ArrayExprStart', text: '['},
// CHECK:STDOUT:               {kind: 'IntegerTypeLiteral', text: 'i32'},
// CHECK:STDOUT:             {kind: 'ArrayExprSemi', text: ';', subtree_size: 3},
// CHECK:STDOUT:             {kind: 'IntegerLiteral', text: '5'},
// CHECK:STDOUT:           {kind: 'ArrayExpr', text: ']', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'PatternBinding', text: ':', subtree_size: 7},
// CHECK:STDOUT:         {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:           {kind: 'ParenExprOrTupleLiteralStart', text: '('},
// CHECK:STDOUT:           {kind: 'IntegerLiteral', text: '8'},
// CHECK:STDOUT:           {kind: 'TupleLiteralComma', text: ','},
// CHECK:STDOUT:           {kind: 'IntegerLiteral', text: '9'},
// CHECK:STDOUT:           {kind: 'TupleLiteralComma', text: ','},
// CHECK:STDOUT:           {kind: 'IntegerLiteral', text: '0x8'},
// CHECK:STDOUT:           {kind: 'TupleLiteralComma', text: ','},
// CHECK:STDOUT:           {kind: 'IntegerLiteral', text: '0b1000'},
// CHECK:STDOUT:           {kind: 'TupleLiteralComma', text: ','},
// CHECK:STDOUT:           {kind: 'IntegerLiteral', text: '39999999999999999993'},
// CHECK:STDOUT:           {kind: 'TupleLiteralComma', text: ','},
// CHECK:STDOUT:         {kind: 'TupleLiteral', text: ')', subtree_size: 12},
// CHECK:STDOUT:       {kind: 'VariableDecl', text: ';', subtree_size: 22},
// CHECK:STDOUT:         {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:           {kind: 'Name', text: 'floats'},
// CHECK:STDOUT:               {kind: 'ArrayExprStart', text: '['},
// CHECK:STDOUT:               {kind: 'FloatingPointTypeLiteral', text: 'f64'},
// CHECK:STDOUT:             {kind: 'ArrayExprSemi', text: ';', subtree_size: 3},
// CHECK:STDOUT:             {kind: 'IntegerLiteral', text: '7'},
// CHECK:STDOUT:           {kind: 'ArrayExpr', text: ']', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'PatternBinding', text: ':', subtree_size: 7},
// CHECK:STDOUT:         {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:           {kind: 'ParenExprOrTupleLiteralStart', text: '('},
// CHECK:STDOUT:           {kind: 'FloatingPointLiteral', text: '0.9'},
// CHECK:STDOUT:           {kind: 'TupleLiteralComma', text: ','},
// CHECK:STDOUT:           {kind: 'FloatingPointLiteral', text: '8.0'},
// CHECK:STDOUT:           {kind: 'TupleLiteralComma', text: ','},
// CHECK:STDOUT:           {kind: 'FloatingPointLiteral', text: '80.0'},
// CHECK:STDOUT:           {kind: 'TupleLiteralComma', text: ','},
// CHECK:STDOUT:           {kind: 'FloatingPointLiteral', text: '1.0e7'},
// CHECK:STDOUT:           {kind: 'TupleLiteralComma', text: ','},
// CHECK:STDOUT:           {kind: 'FloatingPointLiteral', text: '1.0e8'},
// CHECK:STDOUT:           {kind: 'TupleLiteralComma', text: ','},
// CHECK:STDOUT:           {kind: 'FloatingPointLiteral', text: '1.0e-8'},
// CHECK:STDOUT:           {kind: 'TupleLiteralComma', text: ','},
// CHECK:STDOUT:           {kind: 'FloatingPointLiteral', text: '39999999999999999993.0e39999999999999999993'},
// CHECK:STDOUT:           {kind: 'TupleLiteralComma', text: ','},
// CHECK:STDOUT:         {kind: 'TupleLiteral', text: ')', subtree_size: 16},
// CHECK:STDOUT:       {kind: 'VariableDecl', text: ';', subtree_size: 26},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 54},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
