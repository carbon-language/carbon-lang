// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/basics/fail_bracket_recovery.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/basics/fail_bracket_recovery.carbon

// This is a valid parse tree even though the lex errors.
fn F() {
  // CHECK:STDERR: fail_bracket_recovery.carbon:[[@LINE+4]]:11: error: opening symbol without a corresponding closing symbol [UnmatchedOpening]
  // CHECK:STDERR:   var x: ([i32; 3) = (0, 1, 2);
  // CHECK:STDERR:           ^
  // CHECK:STDERR:
  var x: ([i32; 3) = (0, 1, 2);
  // CHECK:STDERR: fail_bracket_recovery.carbon:[[@LINE+12]]:15: error: opening symbol without a corresponding closing symbol [UnmatchedOpening]
  // CHECK:STDERR:   var x: {.y: (} = {.y = ((};
  // CHECK:STDERR:               ^
  // CHECK:STDERR:
  // CHECK:STDERR: fail_bracket_recovery.carbon:[[@LINE+8]]:26: error: opening symbol without a corresponding closing symbol [UnmatchedOpening]
  // CHECK:STDERR:   var x: {.y: (} = {.y = ((};
  // CHECK:STDERR:                          ^
  // CHECK:STDERR:
  // CHECK:STDERR: fail_bracket_recovery.carbon:[[@LINE+4]]:27: error: opening symbol without a corresponding closing symbol [UnmatchedOpening]
  // CHECK:STDERR:   var x: {.y: (} = {.y = ((};
  // CHECK:STDERR:                           ^
  // CHECK:STDERR:
  var x: {.y: (} = {.y = ((};
}

// CHECK:STDOUT: - filename: fail_bracket_recovery.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'F'},
// CHECK:STDOUT:           {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:         {kind: 'TuplePattern', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeParams', text: 'x'},
// CHECK:STDOUT:               {kind: 'ParenExprStart', text: '('},
// CHECK:STDOUT:                   {kind: 'ArrayExprStart', text: '['},
// CHECK:STDOUT:                   {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:                 {kind: 'ArrayExprSemi', text: ';', subtree_size: 3},
// CHECK:STDOUT:                 {kind: 'IntLiteral', text: '3'},
// CHECK:STDOUT:               {kind: 'ArrayExpr', text: ']', subtree_size: 5},
// CHECK:STDOUT:             {kind: 'ParenExpr', text: ')', subtree_size: 7},
// CHECK:STDOUT:           {kind: 'VarBindingPattern', text: ':', subtree_size: 9},
// CHECK:STDOUT:         {kind: 'VariablePattern', text: 'var', subtree_size: 10},
// CHECK:STDOUT:         {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:           {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '0'},
// CHECK:STDOUT:           {kind: 'TupleLiteralComma', text: ','},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '1'},
// CHECK:STDOUT:           {kind: 'TupleLiteralComma', text: ','},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '2'},
// CHECK:STDOUT:         {kind: 'TupleLiteral', text: ')', subtree_size: 7},
// CHECK:STDOUT:       {kind: 'VariableDecl', text: ';', subtree_size: 20},
// CHECK:STDOUT:         {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeParams', text: 'x'},
// CHECK:STDOUT:               {kind: 'StructTypeLiteralStart', text: '{'},
// CHECK:STDOUT:                   {kind: 'IdentifierNameNotBeforeParams', text: 'y'},
// CHECK:STDOUT:                 {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:                   {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:                 {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'StructTypeLiteralField', text: ':', subtree_size: 5},
// CHECK:STDOUT:             {kind: 'StructTypeLiteral', text: '}', subtree_size: 7},
// CHECK:STDOUT:           {kind: 'VarBindingPattern', text: ':', subtree_size: 9},
// CHECK:STDOUT:         {kind: 'VariablePattern', text: 'var', subtree_size: 10},
// CHECK:STDOUT:         {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:           {kind: 'StructLiteralStart', text: '{'},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeParams', text: 'y'},
// CHECK:STDOUT:             {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'ParenExprStart', text: '('},
// CHECK:STDOUT:                 {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:               {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'ParenExpr', text: ')', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'StructLiteralField', text: '=', subtree_size: 7},
// CHECK:STDOUT:         {kind: 'StructLiteral', text: '}', subtree_size: 9},
// CHECK:STDOUT:       {kind: 'VariableDecl', text: ';', subtree_size: 22},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 48},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
