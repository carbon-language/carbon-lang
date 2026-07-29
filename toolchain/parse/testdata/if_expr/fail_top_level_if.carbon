// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/if_expr/fail_top_level_if.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/if_expr/fail_top_level_if.carbon

fn F() {
  // CHECK:STDERR: fail_top_level_if.carbon:[[@LINE+12]]:6: error: expected `(` after `if` [ExpectedParenAfter]
  // CHECK:STDERR:   if true then 1 else 2;
  // CHECK:STDERR:      ^~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_top_level_if.carbon:[[@LINE+8]]:11: error: expected braced code block [ExpectedCodeBlock]
  // CHECK:STDERR:   if true then 1 else 2;
  // CHECK:STDERR:           ^~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_top_level_if.carbon:[[@LINE+4]]:11: error: expected expression [ExpectedExpr]
  // CHECK:STDERR:   if true then 1 else 2;
  // CHECK:STDERR:           ^~~~
  // CHECK:STDERR:
  if true then 1 else 2;
}

// CHECK:STDOUT: - filename: fail_top_level_if.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │   ╭─IfConditionStart 'if' ❌
// CHECK:STDOUT:   │ │   ├─BoolLiteralTrue 'true'
// CHECK:STDOUT:   │ │ ╭─IfCondition 'if' ❌
// CHECK:STDOUT:   │ │ │ ╭─CodeBlockStart 'then' ❌
// CHECK:STDOUT:   │ │ │ │ ╭─InvalidParse 'then' ❌
// CHECK:STDOUT:   │ │ │ ├─ExprStatement ';' ❌
// CHECK:STDOUT:   │ │ ├─CodeBlock 'then' ❌
// CHECK:STDOUT:   │ ├─IfStatement 'if'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
