// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/for/fail_missing_in.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/for/fail_missing_in.carbon

fn foo() {
  // CHECK:STDERR: fail_missing_in.carbon:[[@LINE+4]]:19: error: expected `in` after loop `var` declaration [ExpectedIn]
  // CHECK:STDERR:   for (var x: i32 y) {
  // CHECK:STDERR:                   ^
  // CHECK:STDERR:
  for (var x: i32 y) {
    Print(x);
  }
}

// CHECK:STDOUT: - filename: fail_missing_in.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'foo'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'ForHeaderStart', text: '('},
// CHECK:STDOUT:             {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:                 {kind: 'IdentifierNameNotBeforeParams', text: 'x'},
// CHECK:STDOUT:                 {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:               {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:             {kind: 'VariablePattern', text: 'var', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'ForIn', text: 'var', has_error: yes, subtree_size: 6},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'y'},
// CHECK:STDOUT:         {kind: 'ForHeader', text: ')', subtree_size: 9},
// CHECK:STDOUT:           {kind: 'CodeBlockStart', text: '{'},
// CHECK:STDOUT:                 {kind: 'IdentifierNameExpr', text: 'Print'},
// CHECK:STDOUT:               {kind: 'CallExprStart', text: '(', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'IdentifierNameExpr', text: 'x'},
// CHECK:STDOUT:             {kind: 'CallExpr', text: ')', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'ExprStatement', text: ';', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'CodeBlock', text: '}', subtree_size: 7},
// CHECK:STDOUT:       {kind: 'ForStatement', text: 'for', subtree_size: 17},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 23},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
