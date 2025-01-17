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
  // CHECK:STDERR: fail_square_brackets.carbon:[[@LINE+12]]:6: error: expected `(` after `if` [ExpectedParenAfter]
  // CHECK:STDERR:   if [] {}
  // CHECK:STDERR:      ^
  // CHECK:STDERR:
  // CHECK:STDERR: fail_square_brackets.carbon:[[@LINE+8]]:7: error: expected expression [ExpectedExpr]
  // CHECK:STDERR:   if [] {}
  // CHECK:STDERR:       ^
  // CHECK:STDERR:
  // CHECK:STDERR: fail_square_brackets.carbon:[[@LINE+4]]:7: error: expected `;` in array type [ExpectedArraySemi]
  // CHECK:STDERR:   if [] {}
  // CHECK:STDERR:       ^
  // CHECK:STDERR:
  if [] {}
}

// CHECK:STDOUT: - filename: fail_square_brackets.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'F'},
// CHECK:STDOUT:           {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:         {kind: 'TuplePattern', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'IfConditionStart', text: 'if', has_error: yes},
// CHECK:STDOUT:               {kind: 'ArrayExprStart', text: '['},
// CHECK:STDOUT:               {kind: 'InvalidParse', text: ']', has_error: yes},
// CHECK:STDOUT:             {kind: 'ArrayExprSemi', text: ']', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:           {kind: 'ArrayExpr', text: ']', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:         {kind: 'IfCondition', text: 'if', has_error: yes, subtree_size: 6},
// CHECK:STDOUT:           {kind: 'CodeBlockStart', text: '{'},
// CHECK:STDOUT:         {kind: 'CodeBlock', text: '}', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'IfStatement', text: 'if', subtree_size: 9},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 15},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
