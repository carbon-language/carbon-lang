// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// This is a valid parse tree even though the lex errors.
fn F() {
  // CHECK:STDERR: fail_bracket_recovery.carbon:[[@LINE+4]]:11: ERROR: Opening symbol without a corresponding closing symbol.
  // CHECK:STDERR:   var x: ([i32; 3) = (0, 1, 2);
  // CHECK:STDERR:           ^
  // CHECK:STDERR:
  var x: ([i32; 3) = (0, 1, 2);
  // CHECK:STDERR: fail_bracket_recovery.carbon:[[@LINE+11]]:15: ERROR: Opening symbol without a corresponding closing symbol.
  // CHECK:STDERR:   var x: {.y: (} = {.y = ((};
  // CHECK:STDERR:               ^
  // CHECK:STDERR:
  // CHECK:STDERR: fail_bracket_recovery.carbon:[[@LINE+7]]:26: ERROR: Opening symbol without a corresponding closing symbol.
  // CHECK:STDERR:   var x: {.y: (} = {.y = ((};
  // CHECK:STDERR:                          ^
  // CHECK:STDERR:
  // CHECK:STDERR: fail_bracket_recovery.carbon:[[@LINE+3]]:27: ERROR: Opening symbol without a corresponding closing symbol.
  // CHECK:STDERR:   var x: {.y: (} = {.y = ((};
  // CHECK:STDERR:                           ^
  var x: {.y: (} = {.y = ((};
}

// CHECK:STDOUT: - filename: fail_bracket_recovery.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'F'},
// CHECK:STDOUT:           {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:         {kind: 'TuplePattern', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:           {kind: 'IdentifierName', text: 'x'},
// CHECK:STDOUT:             {kind: 'ParenExprStart', text: '('},
// CHECK:STDOUT:                 {kind: 'ArrayExprStart', text: '['},
// CHECK:STDOUT:                 {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:               {kind: 'ArrayExprSemi', text: ';', subtree_size: 3},
// CHECK:STDOUT:               {kind: 'IntLiteral', text: '3'},
// CHECK:STDOUT:             {kind: 'ArrayExpr', text: ']', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'ParenExpr', text: ')', subtree_size: 7},
// CHECK:STDOUT:         {kind: 'BindingPattern', text: ':', subtree_size: 9},
// CHECK:STDOUT:         {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:           {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '0'},
// CHECK:STDOUT:           {kind: 'TupleLiteralComma', text: ','},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '1'},
// CHECK:STDOUT:           {kind: 'TupleLiteralComma', text: ','},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '2'},
// CHECK:STDOUT:         {kind: 'TupleLiteral', text: ')', subtree_size: 7},
// CHECK:STDOUT:       {kind: 'VariableDecl', text: ';', subtree_size: 19},
// CHECK:STDOUT:         {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:           {kind: 'IdentifierName', text: 'x'},
// CHECK:STDOUT:             {kind: 'StructTypeLiteralStart', text: '{'},
// CHECK:STDOUT:                 {kind: 'IdentifierName', text: 'y'},
// CHECK:STDOUT:               {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:                 {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:               {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'StructTypeField', text: ':', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'StructTypeLiteral', text: '}', subtree_size: 7},
// CHECK:STDOUT:         {kind: 'BindingPattern', text: ':', subtree_size: 9},
// CHECK:STDOUT:         {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:           {kind: 'StructLiteralStart', text: '{'},
// CHECK:STDOUT:               {kind: 'IdentifierName', text: 'y'},
// CHECK:STDOUT:             {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'ParenExprStart', text: '('},
// CHECK:STDOUT:                 {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:               {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'ParenExpr', text: ')', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'StructField', text: '=', subtree_size: 7},
// CHECK:STDOUT:         {kind: 'StructLiteral', text: '}', subtree_size: 9},
// CHECK:STDOUT:       {kind: 'VariableDecl', text: ';', subtree_size: 21},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 46},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
