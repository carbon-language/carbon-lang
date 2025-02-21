// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/pointer/pointer_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/pointer/pointer_type.carbon

fn F(p: i32*) -> i32* {
  f32* + 0;
  1 * bool* * 2;
  return 3 * i32* * *p;
}

var T: type = if true then i32* else f64*;

// CHECK:STDOUT: - filename: pointer_type.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'F'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeParams', text: 'p'},
// CHECK:STDOUT:               {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:             {kind: 'PostfixOperatorStar', text: '*', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'LetBindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 6},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'PostfixOperatorStar', text: '*', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'ReturnType', text: '->', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 12},
// CHECK:STDOUT:             {kind: 'FloatTypeLiteral', text: 'f32'},
// CHECK:STDOUT:           {kind: 'PostfixOperatorStar', text: '*', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '0'},
// CHECK:STDOUT:         {kind: 'InfixOperatorPlus', text: '+', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 5},
// CHECK:STDOUT:             {kind: 'IntLiteral', text: '1'},
// CHECK:STDOUT:               {kind: 'BoolTypeLiteral', text: 'bool'},
// CHECK:STDOUT:             {kind: 'PostfixOperatorStar', text: '*', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'InfixOperatorStar', text: '*', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '2'},
// CHECK:STDOUT:         {kind: 'InfixOperatorStar', text: '*', subtree_size: 6},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 7},
// CHECK:STDOUT:         {kind: 'ReturnStatementStart', text: 'return'},
// CHECK:STDOUT:             {kind: 'IntLiteral', text: '3'},
// CHECK:STDOUT:               {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:             {kind: 'PostfixOperatorStar', text: '*', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'InfixOperatorStar', text: '*', subtree_size: 4},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'p'},
// CHECK:STDOUT:           {kind: 'PrefixOperatorStar', text: '*', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'InfixOperatorStar', text: '*', subtree_size: 7},
// CHECK:STDOUT:       {kind: 'ReturnStatement', text: ';', subtree_size: 9},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 34},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'T'},
// CHECK:STDOUT:           {kind: 'TypeTypeLiteral', text: 'type'},
// CHECK:STDOUT:         {kind: 'VarBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'VariablePattern', text: 'var', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:           {kind: 'BoolLiteralTrue', text: 'true'},
// CHECK:STDOUT:         {kind: 'IfExprIf', text: 'if', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'PostfixOperatorStar', text: '*', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'IfExprThen', text: 'then', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'FloatTypeLiteral', text: 'f64'},
// CHECK:STDOUT:         {kind: 'PostfixOperatorStar', text: '*', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'IfExprElse', text: 'else', subtree_size: 8},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: ';', subtree_size: 15},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
