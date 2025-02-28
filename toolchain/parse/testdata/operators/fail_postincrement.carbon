// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/operators/fail_postincrement.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/operators/fail_postincrement.carbon

fn F() {
  var n: i32 = 0;
  // TODO: It'd be nice to produce a custom diagnostic here.
  // CHECK:STDERR: fail_postincrement.carbon:[[@LINE+4]]:4: error: expected `;` after expression statement [ExpectedExprSemi]
  // CHECK:STDERR:   n++;
  // CHECK:STDERR:    ^~
  // CHECK:STDERR:
  n++;
  // CHECK:STDERR: fail_postincrement.carbon:[[@LINE+4]]:4: error: expected `;` after expression statement [ExpectedExprSemi]
  // CHECK:STDERR:   n--;
  // CHECK:STDERR:    ^~
  // CHECK:STDERR:
  n--;
}

// CHECK:STDOUT: - filename: fail_postincrement.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'F'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeParams', text: 'n'},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'VarBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'VariablePattern', text: 'var', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'IntLiteral', text: '0'},
// CHECK:STDOUT:       {kind: 'VariableDecl', text: ';', subtree_size: 8},
// CHECK:STDOUT:         {kind: 'IdentifierNameExpr', text: 'n'},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:         {kind: 'IdentifierNameExpr', text: 'n'},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 18},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
