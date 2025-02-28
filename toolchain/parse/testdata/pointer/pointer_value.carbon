// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/pointer/pointer_value.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/pointer/pointer_value.carbon

fn F() -> i32 {
  var n: i32 = 0;
  var p: i32* = &n;
  var q: i32** = &p;
  return **q;
}

// CHECK:STDOUT: - filename: pointer_value.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'F'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'ReturnType', text: '->', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 7},
// CHECK:STDOUT:         {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeParams', text: 'n'},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'VarBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'VariablePattern', text: 'var', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'IntLiteral', text: '0'},
// CHECK:STDOUT:       {kind: 'VariableDecl', text: ';', subtree_size: 8},
// CHECK:STDOUT:         {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeParams', text: 'p'},
// CHECK:STDOUT:               {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:             {kind: 'PostfixOperatorStar', text: '*', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'VarBindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'VariablePattern', text: 'var', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'n'},
// CHECK:STDOUT:         {kind: 'PrefixOperatorAmp', text: '&', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'VariableDecl', text: ';', subtree_size: 10},
// CHECK:STDOUT:         {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeParams', text: 'q'},
// CHECK:STDOUT:                 {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:               {kind: 'PostfixOperatorStar', text: '*', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'PostfixOperatorStar', text: '*', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'VarBindingPattern', text: ':', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'VariablePattern', text: 'var', subtree_size: 6},
// CHECK:STDOUT:         {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'p'},
// CHECK:STDOUT:         {kind: 'PrefixOperatorAmp', text: '&', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'VariableDecl', text: ';', subtree_size: 11},
// CHECK:STDOUT:         {kind: 'ReturnStatementStart', text: 'return'},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'q'},
// CHECK:STDOUT:           {kind: 'PrefixOperatorStar', text: '*', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'PrefixOperatorStar', text: '*', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ReturnStatement', text: ';', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 42},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
