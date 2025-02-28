// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/operators/fixity_in_call.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/operators/fixity_in_call.carbon

fn F() {
  G(i32*, n * n);
}

// CHECK:STDOUT: - filename: fixity_in_call.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'F'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'G'},
// CHECK:STDOUT:           {kind: 'CallExprStart', text: '(', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'PostfixOperatorStar', text: '*', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'CallExprComma', text: ','},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'n'},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'n'},
// CHECK:STDOUT:           {kind: 'InfixOperatorStar', text: '*', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'CallExpr', text: ')', subtree_size: 9},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 10},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 16},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
