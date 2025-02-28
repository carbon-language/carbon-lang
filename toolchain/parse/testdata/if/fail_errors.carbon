// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/if/fail_errors.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/if/fail_errors.carbon

fn F() {
  // CHECK:STDERR: fail_errors.carbon:[[@LINE+4]]:6: error: expected `(` after `if` [ExpectedParenAfter]
  // CHECK:STDERR:   if a {}
  // CHECK:STDERR:      ^
  // CHECK:STDERR:
  if a {}
  // CHECK:STDERR: fail_errors.carbon:[[@LINE+4]]:7: error: expected expression [ExpectedExpr]
  // CHECK:STDERR:   if () {}
  // CHECK:STDERR:       ^
  // CHECK:STDERR:
  if () {}
  // CHECK:STDERR: fail_errors.carbon:[[@LINE+4]]:9: error: unexpected tokens before `)` [ExpectedCloseSymbol]
  // CHECK:STDERR:   if (b c) {}
  // CHECK:STDERR:         ^
  // CHECK:STDERR:
  if (b c) {}
  if (d)
// CHECK:STDERR: fail_errors.carbon:[[@LINE+8]]:1: error: expected braced code block [ExpectedCodeBlock]
// CHECK:STDERR: }
// CHECK:STDERR: ^
// CHECK:STDERR:
// CHECK:STDERR: fail_errors.carbon:[[@LINE+4]]:1: error: expected expression [ExpectedExpr]
// CHECK:STDERR: }
// CHECK:STDERR: ^
// CHECK:STDERR:
}

// CHECK:STDOUT: - filename: fail_errors.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'F'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'IfConditionStart', text: 'if', has_error: yes},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'a'},
// CHECK:STDOUT:         {kind: 'IfCondition', text: 'if', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:           {kind: 'CodeBlockStart', text: '{'},
// CHECK:STDOUT:         {kind: 'CodeBlock', text: '}', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'IfStatement', text: 'if', subtree_size: 6},
// CHECK:STDOUT:           {kind: 'IfConditionStart', text: '('},
// CHECK:STDOUT:           {kind: 'InvalidParse', text: ')', has_error: yes},
// CHECK:STDOUT:         {kind: 'IfCondition', text: ')', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:           {kind: 'CodeBlockStart', text: '{'},
// CHECK:STDOUT:         {kind: 'CodeBlock', text: '}', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'IfStatement', text: 'if', subtree_size: 6},
// CHECK:STDOUT:           {kind: 'IfConditionStart', text: '('},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'b'},
// CHECK:STDOUT:         {kind: 'IfCondition', text: ')', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:           {kind: 'CodeBlockStart', text: '{'},
// CHECK:STDOUT:         {kind: 'CodeBlock', text: '}', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'IfStatement', text: 'if', subtree_size: 6},
// CHECK:STDOUT:           {kind: 'IfConditionStart', text: '('},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'd'},
// CHECK:STDOUT:         {kind: 'IfCondition', text: ')', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'CodeBlockStart', text: '}', has_error: yes},
// CHECK:STDOUT:             {kind: 'InvalidParse', text: '}', has_error: yes},
// CHECK:STDOUT:           {kind: 'ExprStatement', text: ')', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:         {kind: 'CodeBlock', text: '}', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:       {kind: 'IfStatement', text: 'if', subtree_size: 8},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 32},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
