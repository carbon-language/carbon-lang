// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/for/fail_missing_cond.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/for/fail_missing_cond.carbon

fn F() {
  // CHECK:STDERR: fail_missing_cond.carbon:[[@LINE+4]]:7: error: expected `(` after `for` [ExpectedParenAfter]
  // CHECK:STDERR:   for {
  // CHECK:STDERR:       ^
  // CHECK:STDERR:
  for {
  }
// CHECK:STDERR: fail_missing_cond.carbon:[[@LINE+16]]:1: error: expected `in` after loop pattern [ExpectedIn]
// CHECK:STDERR: }
// CHECK:STDERR: ^
// CHECK:STDERR:
// CHECK:STDERR: fail_missing_cond.carbon:[[@LINE+12]]:1: error: expected expression [ExpectedExpr]
// CHECK:STDERR: }
// CHECK:STDERR: ^
// CHECK:STDERR:
// CHECK:STDERR: fail_missing_cond.carbon:[[@LINE+8]]:1: error: expected braced code block [ExpectedCodeBlock]
// CHECK:STDERR: }
// CHECK:STDERR: ^
// CHECK:STDERR:
// CHECK:STDERR: fail_missing_cond.carbon:[[@LINE+4]]:1: error: expected expression [ExpectedExpr]
// CHECK:STDERR: }
// CHECK:STDERR: ^
// CHECK:STDERR:
}

// CHECK:STDOUT: - filename: fail_missing_cond.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │   ╭─ForHeaderStart 'for' ❌
// CHECK:STDOUT:   │ │   │   ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ │   │ ╭─StructLiteral '}'
// CHECK:STDOUT:   │ │   ├─ForIn 'for' ❌
// CHECK:STDOUT:   │ │   ├─InvalidParse '}' ❌
// CHECK:STDOUT:   │ │ ╭─ForHeader 'for' ❌
// CHECK:STDOUT:   │ │ │ ╭─CodeBlockStart '}' ❌
// CHECK:STDOUT:   │ │ │ │ ╭─InvalidParse '}' ❌
// CHECK:STDOUT:   │ │ │ ├─ExprStatement '}' ❌
// CHECK:STDOUT:   │ │ ├─CodeBlock '}' ❌
// CHECK:STDOUT:   │ ├─ForStatement 'for'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
