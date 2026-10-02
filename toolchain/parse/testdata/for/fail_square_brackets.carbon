// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/for/fail_square_brackets.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/for/fail_square_brackets.carbon

fn F() {
  // CHECK:STDERR: fail_square_brackets.carbon:[[@LINE+10]]:7: error: expected `(` after `for` [ExpectedParenAfter]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_square_brackets.carbon:[[@LINE+8]]:7: error: expected pattern [ExpectedPattern]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_square_brackets.carbon:[[@LINE+6]]:7: error: expected expression [ExpectedExpr]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_square_brackets.carbon:[[@LINE+4]]:7: error: expected braced code block [ExpectedCodeBlock]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_square_brackets.carbon:[[@LINE+2]]:7: error: expected expression [ExpectedExpr]
  // CHECK:STDERR:
  for [] {
  }
}

// CHECK:STDOUT: - filename: fail_square_brackets.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │   ╭─ForHeaderStart 'for' has_error
// CHECK:STDOUT:   │ │   │ ╭─InvalidParse '[' has_error
// CHECK:STDOUT:   │ │   ├─ForIn 'for' has_error
// CHECK:STDOUT:   │ │   ├─InvalidParse '[' has_error
// CHECK:STDOUT:   │ │ ╭─ForHeader 'for' has_error
// CHECK:STDOUT:   │ │ │ ╭─CodeBlockStart '[' has_error
// CHECK:STDOUT:   │ │ │ │ ╭─InvalidParse '[' has_error
// CHECK:STDOUT:   │ │ │ ├─ExprStatement '}' has_error
// CHECK:STDOUT:   │ │ ├─CodeBlock '[' has_error
// CHECK:STDOUT:   │ ├─ForStatement 'for'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
