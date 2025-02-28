// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/operators/precedence_as.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/operators/precedence_as.carbon

fn F(n: i32) {
  // Type operators and unary operators are higher precedence than `as`.
  -n as const i32;
  &n as i32*;
  ^n as i32;
  *n as i32;
}

// CHECK:STDOUT: - filename: precedence_as.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'F'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeParams', text: 'n'},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 8},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'n'},
// CHECK:STDOUT:           {kind: 'PrefixOperatorMinus', text: '-', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'PrefixOperatorConst', text: 'const', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'InfixOperatorAs', text: 'as', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 6},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'n'},
// CHECK:STDOUT:           {kind: 'PrefixOperatorAmp', text: '&', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'PostfixOperatorStar', text: '*', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'InfixOperatorAs', text: 'as', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 6},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'n'},
// CHECK:STDOUT:           {kind: 'PrefixOperatorCaret', text: '^', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'InfixOperatorAs', text: 'as', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 5},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'n'},
// CHECK:STDOUT:           {kind: 'PrefixOperatorStar', text: '*', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'InfixOperatorAs', text: 'as', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 31},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
