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
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │   ╭─IfConditionStart 'if' ❌
// CHECK:STDOUT:   │ │   ├─InvalidParse '[' ❌
// CHECK:STDOUT:   │ │ ╭─IfCondition 'if' ❌
// CHECK:STDOUT:   │ │ │ ╭─CodeBlockStart '[' ❌
// CHECK:STDOUT:   │ │ │ │ ╭─InvalidParse '[' ❌
// CHECK:STDOUT:   │ │ │ ├─ExprStatement '}' ❌
// CHECK:STDOUT:   │ │ ├─CodeBlock '[' ❌
// CHECK:STDOUT:   │ ├─IfStatement 'if'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
