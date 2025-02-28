// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/function/definition/fail_identifier_in_statements.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/function/definition/fail_identifier_in_statements.carbon

fn F() {
  // Note: this might become valid depending on the expression syntax. This test
  // shouldn't be taken as a sign it should remain invalid.
  bar
// CHECK:STDERR: fail_identifier_in_statements.carbon:[[@LINE+4]]:1: error: expected `;` after expression statement [ExpectedExprSemi]
// CHECK:STDERR: }
// CHECK:STDERR: ^
// CHECK:STDERR:
}

// CHECK:STDOUT: - filename: fail_identifier_in_statements.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'F'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'IdentifierNameExpr', text: 'bar'},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: 'bar', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 8},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
