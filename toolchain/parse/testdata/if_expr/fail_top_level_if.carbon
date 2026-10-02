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
  // CHECK:STDERR: fail_top_level_if.carbon:[[@LINE+6]]:6-9: error: expected `(` after `if` [ExpectedParenAfter]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_top_level_if.carbon:[[@LINE+4]]:11-14: error: expected braced code block [ExpectedCodeBlock]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_top_level_if.carbon:[[@LINE+2]]:11-14: error: expected expression [ExpectedExpr]
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
// CHECK:STDOUT:   │ │   ╭─IfConditionStart 'if' has_error
// CHECK:STDOUT:   │ │   ├─BoolLiteralTrue 'true'
// CHECK:STDOUT:   │ │ ╭─IfCondition 'if' has_error
// CHECK:STDOUT:   │ │ │ ╭─CodeBlockStart 'then' has_error
// CHECK:STDOUT:   │ │ │ │ ╭─InvalidParse 'then' has_error
// CHECK:STDOUT:   │ │ │ ├─ExprStatement ';' has_error
// CHECK:STDOUT:   │ │ ├─CodeBlock 'then' has_error
// CHECK:STDOUT:   │ ├─IfStatement 'if'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
