// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/if_expr/fail_else_missing.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/if_expr/fail_else_missing.carbon

fn F() {
  // CHECK:STDERR: fail_else_missing.carbon:[[@LINE+4]]:30: error: expected `else` after `if ... then ...` [ExpectedElseAfterIf]
  // CHECK:STDERR:   var n: i32 = if true then 1;
  // CHECK:STDERR:                              ^
  // CHECK:STDERR:
  var n: i32 = if true then 1;
}

// CHECK:STDOUT: - filename: fail_else_missing.carbon
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
// CHECK:STDOUT:             {kind: 'BoolLiteralTrue', text: 'true'},
// CHECK:STDOUT:           {kind: 'IfExprIf', text: 'if', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'IntLiteral', text: '1'},
// CHECK:STDOUT:           {kind: 'IfExprThen', text: 'then', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'InvalidParse', text: ';', has_error: yes},
// CHECK:STDOUT:         {kind: 'IfExprElse', text: 'if', has_error: yes, subtree_size: 6},
// CHECK:STDOUT:       {kind: 'VariableDecl', text: ';', subtree_size: 13},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 19},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
