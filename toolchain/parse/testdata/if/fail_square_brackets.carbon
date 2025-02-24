// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/if/fail_square_brackets.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/if/fail_square_brackets.carbon

fn F() {
  // CHECK:STDERR: fail_square_brackets.carbon:[[@LINE+16]]:6: error: expected `(` after `if` [ExpectedParenAfter]
  // CHECK:STDERR:   if [] {}
  // CHECK:STDERR:      ^
  // CHECK:STDERR:
  // CHECK:STDERR: fail_square_brackets.carbon:[[@LINE+12]]:6: error: expected expression [ExpectedExpr]
  // CHECK:STDERR:   if [] {}
  // CHECK:STDERR:      ^
  // CHECK:STDERR:
  // CHECK:STDERR: fail_square_brackets.carbon:[[@LINE+8]]:6: error: expected braced code block [ExpectedCodeBlock]
  // CHECK:STDERR:   if [] {}
  // CHECK:STDERR:      ^
  // CHECK:STDERR:
  // CHECK:STDERR: fail_square_brackets.carbon:[[@LINE+4]]:6: error: expected expression [ExpectedExpr]
  // CHECK:STDERR:   if [] {}
  // CHECK:STDERR:      ^
  // CHECK:STDERR:
  if [] {}
}

// CHECK:STDOUT: - filename: fail_square_brackets.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'F'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'IfConditionStart', text: 'if', has_error: yes},
// CHECK:STDOUT:           {kind: 'InvalidParse', text: '[', has_error: yes},
// CHECK:STDOUT:         {kind: 'IfCondition', text: 'if', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:           {kind: 'CodeBlockStart', text: '[', has_error: yes},
// CHECK:STDOUT:             {kind: 'InvalidParse', text: '[', has_error: yes},
// CHECK:STDOUT:           {kind: 'ExprStatement', text: '}', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:         {kind: 'CodeBlock', text: '[', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:       {kind: 'IfStatement', text: 'if', subtree_size: 8},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 14},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
