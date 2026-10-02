// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// TODO: This should have an error.
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/if/fail_unbraced.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/if/fail_unbraced.carbon

fn F() {
  if (a)
    // CHECK:STDERR: fail_unbraced.carbon:[[@LINE+2]]:5-6: error: expected braced code block [ExpectedCodeBlock]
    // CHECK:STDERR:
    if (b)
      // CHECK:STDERR: fail_unbraced.carbon:[[@LINE+2]]:7-8: error: expected braced code block [ExpectedCodeBlock]
      // CHECK:STDERR:
      if (c)
        // CHECK:STDERR: fail_unbraced.carbon:[[@LINE+2]]:9: error: expected braced code block [ExpectedCodeBlock]
        // CHECK:STDERR:
        d;
}

// CHECK:STDOUT: - filename: fail_unbraced.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │   ╭─IfConditionStart '('
// CHECK:STDOUT:   │ │   ├─IdentifierNameExpr 'a'
// CHECK:STDOUT:   │ │ ╭─IfCondition ')'
// CHECK:STDOUT:   │ │ │ ╭─CodeBlockStart 'if' has_error
// CHECK:STDOUT:   │ │ │ │   ╭─IfConditionStart '('
// CHECK:STDOUT:   │ │ │ │   ├─IdentifierNameExpr 'b'
// CHECK:STDOUT:   │ │ │ │ ╭─IfCondition ')'
// CHECK:STDOUT:   │ │ │ │ │ ╭─CodeBlockStart 'if' has_error
// CHECK:STDOUT:   │ │ │ │ │ │   ╭─IfConditionStart '('
// CHECK:STDOUT:   │ │ │ │ │ │   ├─IdentifierNameExpr 'c'
// CHECK:STDOUT:   │ │ │ │ │ │ ╭─IfCondition ')'
// CHECK:STDOUT:   │ │ │ │ │ │ │ ╭─CodeBlockStart 'd' has_error
// CHECK:STDOUT:   │ │ │ │ │ │ │ │ ╭─IdentifierNameExpr 'd'
// CHECK:STDOUT:   │ │ │ │ │ │ │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │ │ │ │ │ ├─CodeBlock 'd' has_error
// CHECK:STDOUT:   │ │ │ │ │ ├─IfStatement 'if'
// CHECK:STDOUT:   │ │ │ │ ├─CodeBlock 'if' has_error
// CHECK:STDOUT:   │ │ │ ├─IfStatement 'if'
// CHECK:STDOUT:   │ │ ├─CodeBlock 'if' has_error
// CHECK:STDOUT:   │ ├─IfStatement 'if'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
