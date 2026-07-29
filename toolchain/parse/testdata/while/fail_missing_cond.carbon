// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/while/fail_missing_cond.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/while/fail_missing_cond.carbon

fn F() {
  // CHECK:STDERR: fail_missing_cond.carbon:[[@LINE+4]]:9: error: expected `(` after `while` [ExpectedParenAfter]
  // CHECK:STDERR:   while {
  // CHECK:STDERR:         ^
  // CHECK:STDERR:
  while {
  }
}

// CHECK:STDOUT: - filename: fail_missing_cond.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │   ╭─WhileConditionStart 'while' ❌
// CHECK:STDOUT:   │ │   ├─InvalidParse '{' ❌
// CHECK:STDOUT:   │ │ ╭─WhileCondition 'while' ❌
// CHECK:STDOUT:   │ │ │ ╭─CodeBlockStart '{'
// CHECK:STDOUT:   │ │ ├─CodeBlock '}'
// CHECK:STDOUT:   │ ├─WhileStatement 'while'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
