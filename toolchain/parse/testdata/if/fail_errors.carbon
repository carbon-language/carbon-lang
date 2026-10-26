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
  // CHECK:STDERR: fail_errors.carbon:[[@LINE+2]]:6: error: expected `(` after `if` [ExpectedParenAfter]
  // CHECK:STDERR:
  if a {}
  // CHECK:STDERR: fail_errors.carbon:[[@LINE+2]]:7: error: expected expression [ExpectedExpr]
  // CHECK:STDERR:
  if () {}
  // CHECK:STDERR: fail_errors.carbon:[[@LINE+2]]:9: error: unexpected tokens before `)` [ExpectedCloseSymbol]
  // CHECK:STDERR:
  if (b c) {}
  if (d)
// CHECK:STDERR: fail_errors.carbon:[[@LINE+4]]:1: error: expected braced code block [ExpectedCodeBlock]
// CHECK:STDERR:
// CHECK:STDERR: fail_errors.carbon:[[@LINE+2]]:1: error: expected expression [ExpectedExpr]
// CHECK:STDERR:
}

// CHECK:STDOUT: - filename: fail_errors.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │   ╭─IfConditionStart 'if' has_error
// CHECK:STDOUT:   │ │   ├─IdentifierNameExpr 'a'
// CHECK:STDOUT:   │ │ ╭─IfCondition 'if' has_error
// CHECK:STDOUT:   │ │ │ ╭─CodeBlockStart '{'
// CHECK:STDOUT:   │ │ ├─CodeBlock '}'
// CHECK:STDOUT:   │ ├─IfStatement 'if'
// CHECK:STDOUT:   │ │   ╭─IfConditionStart '('
// CHECK:STDOUT:   │ │   ├─InvalidParse ')' has_error
// CHECK:STDOUT:   │ │ ╭─IfCondition ')' has_error
// CHECK:STDOUT:   │ │ │ ╭─CodeBlockStart '{'
// CHECK:STDOUT:   │ │ ├─CodeBlock '}'
// CHECK:STDOUT:   │ ├─IfStatement 'if'
// CHECK:STDOUT:   │ │   ╭─IfConditionStart '('
// CHECK:STDOUT:   │ │   ├─IdentifierNameExpr 'b'
// CHECK:STDOUT:   │ │ ╭─IfCondition ')' has_error
// CHECK:STDOUT:   │ │ │ ╭─CodeBlockStart '{'
// CHECK:STDOUT:   │ │ ├─CodeBlock '}'
// CHECK:STDOUT:   │ ├─IfStatement 'if'
// CHECK:STDOUT:   │ │   ╭─IfConditionStart '('
// CHECK:STDOUT:   │ │   ├─IdentifierNameExpr 'd'
// CHECK:STDOUT:   │ │ ╭─IfCondition ')'
// CHECK:STDOUT:   │ │ │ ╭─CodeBlockStart '}' has_error
// CHECK:STDOUT:   │ │ │ │ ╭─InvalidParse '}' has_error
// CHECK:STDOUT:   │ │ │ ├─ExprStatement ')' has_error
// CHECK:STDOUT:   │ │ ├─CodeBlock '}' has_error
// CHECK:STDOUT:   │ ├─IfStatement 'if'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
