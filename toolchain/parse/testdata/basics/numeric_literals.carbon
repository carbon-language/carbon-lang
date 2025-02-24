// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/basics/numeric_literals.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/basics/numeric_literals.carbon

fn F() {
  // 8 and 9 trigger special behavior in APInt when mishandling signed versus
  // unsigned, so we pay extra attention to those.
  var ints: array(i32, 5) = (
    8,
    9,
    0x8,
    0b1000,
    39999999999999999993,
  );
  var floats: array(f64, 7) = (
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
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'F'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeParams', text: 'ints'},
// CHECK:STDOUT:               {kind: 'ArrayExprKeyword', text: 'array'},
// CHECK:STDOUT:               {kind: 'ArrayExprOpenParen', text: '('},
// CHECK:STDOUT:               {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:               {kind: 'ArrayExprComma', text: ','},
// CHECK:STDOUT:               {kind: 'IntLiteral', text: '5'},
// CHECK:STDOUT:             {kind: 'ArrayExpr', text: ')', subtree_size: 6},
// CHECK:STDOUT:           {kind: 'VarBindingPattern', text: ':', subtree_size: 8},
// CHECK:STDOUT:         {kind: 'VariablePattern', text: 'var', subtree_size: 9},
// CHECK:STDOUT:         {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:           {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '8'},
// CHECK:STDOUT:           {kind: 'TupleLiteralComma', text: ','},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '9'},
// CHECK:STDOUT:           {kind: 'TupleLiteralComma', text: ','},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '0x8'},
// CHECK:STDOUT:           {kind: 'TupleLiteralComma', text: ','},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '0b1000'},
// CHECK:STDOUT:           {kind: 'TupleLiteralComma', text: ','},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '39999999999999999993'},
// CHECK:STDOUT:           {kind: 'TupleLiteralComma', text: ','},
// CHECK:STDOUT:         {kind: 'TupleLiteral', text: ')', subtree_size: 12},
// CHECK:STDOUT:       {kind: 'VariableDecl', text: ';', subtree_size: 24},
// CHECK:STDOUT:         {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeParams', text: 'floats'},
// CHECK:STDOUT:               {kind: 'ArrayExprKeyword', text: 'array'},
// CHECK:STDOUT:               {kind: 'ArrayExprOpenParen', text: '('},
// CHECK:STDOUT:               {kind: 'FloatTypeLiteral', text: 'f64'},
// CHECK:STDOUT:               {kind: 'ArrayExprComma', text: ','},
// CHECK:STDOUT:               {kind: 'IntLiteral', text: '7'},
// CHECK:STDOUT:             {kind: 'ArrayExpr', text: ')', subtree_size: 6},
// CHECK:STDOUT:           {kind: 'VarBindingPattern', text: ':', subtree_size: 8},
// CHECK:STDOUT:         {kind: 'VariablePattern', text: 'var', subtree_size: 9},
// CHECK:STDOUT:         {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:           {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:           {kind: 'RealLiteral', text: '0.9'},
// CHECK:STDOUT:           {kind: 'TupleLiteralComma', text: ','},
// CHECK:STDOUT:           {kind: 'RealLiteral', text: '8.0'},
// CHECK:STDOUT:           {kind: 'TupleLiteralComma', text: ','},
// CHECK:STDOUT:           {kind: 'RealLiteral', text: '80.0'},
// CHECK:STDOUT:           {kind: 'TupleLiteralComma', text: ','},
// CHECK:STDOUT:           {kind: 'RealLiteral', text: '1.0e7'},
// CHECK:STDOUT:           {kind: 'TupleLiteralComma', text: ','},
// CHECK:STDOUT:           {kind: 'RealLiteral', text: '1.0e8'},
// CHECK:STDOUT:           {kind: 'TupleLiteralComma', text: ','},
// CHECK:STDOUT:           {kind: 'RealLiteral', text: '1.0e-8'},
// CHECK:STDOUT:           {kind: 'TupleLiteralComma', text: ','},
// CHECK:STDOUT:           {kind: 'RealLiteral', text: '39999999999999999993.0e39999999999999999993'},
// CHECK:STDOUT:           {kind: 'TupleLiteralComma', text: ','},
// CHECK:STDOUT:         {kind: 'TupleLiteral', text: ')', subtree_size: 16},
// CHECK:STDOUT:       {kind: 'VariableDecl', text: ';', subtree_size: 28},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 58},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
