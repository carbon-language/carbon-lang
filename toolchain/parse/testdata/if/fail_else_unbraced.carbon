// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/if/fail_else_unbraced.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/if/fail_else_unbraced.carbon

fn F() {
  if (a)
    // CHECK:STDERR: fail_else_unbraced.carbon:[[@LINE+2]]:5-6: error: expected braced code block [ExpectedCodeBlock]
    // CHECK:STDERR:
    if (b)
      // CHECK:STDERR: fail_else_unbraced.carbon:[[@LINE+2]]:7: error: expected braced code block [ExpectedCodeBlock]
      // CHECK:STDERR:
      c;
    else
      // CHECK:STDERR: fail_else_unbraced.carbon:[[@LINE+2]]:7: error: expected braced code block [ExpectedCodeBlock]
      // CHECK:STDERR:
      d;
  else
    // CHECK:STDERR: fail_else_unbraced.carbon:[[@LINE+2]]:5: error: expected braced code block [ExpectedCodeBlock]
    // CHECK:STDERR:
    e;
  if (x) { f; }
  else if (x) { g; }
  else { h; }
}

// CHECK:STDOUT: - filename: fail_else_unbraced.carbon
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
// CHECK:STDOUT:   │ │ │ │ │ ╭─CodeBlockStart 'c' has_error
// CHECK:STDOUT:   │ │ │ │ │ │ ╭─IdentifierNameExpr 'c'
// CHECK:STDOUT:   │ │ │ │ │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │ │ │ ├─CodeBlock 'c' has_error
// CHECK:STDOUT:   │ │ │ │ ├─IfStatementElse 'else'
// CHECK:STDOUT:   │ │ │ │ │ ╭─CodeBlockStart 'd' has_error
// CHECK:STDOUT:   │ │ │ │ │ │ ╭─IdentifierNameExpr 'd'
// CHECK:STDOUT:   │ │ │ │ │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │ │ │ ├─CodeBlock 'd' has_error
// CHECK:STDOUT:   │ │ │ ├─IfStatement 'if'
// CHECK:STDOUT:   │ │ ├─CodeBlock 'if' has_error
// CHECK:STDOUT:   │ │ ├─IfStatementElse 'else'
// CHECK:STDOUT:   │ │ │ ╭─CodeBlockStart 'e' has_error
// CHECK:STDOUT:   │ │ │ │ ╭─IdentifierNameExpr 'e'
// CHECK:STDOUT:   │ │ │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │ ├─CodeBlock 'e' has_error
// CHECK:STDOUT:   │ ├─IfStatement 'if'
// CHECK:STDOUT:   │ │   ╭─IfConditionStart '('
// CHECK:STDOUT:   │ │   ├─IdentifierNameExpr 'x'
// CHECK:STDOUT:   │ │ ╭─IfCondition ')'
// CHECK:STDOUT:   │ │ │ ╭─CodeBlockStart '{'
// CHECK:STDOUT:   │ │ │ │ ╭─IdentifierNameExpr 'f'
// CHECK:STDOUT:   │ │ │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │ ├─CodeBlock '}'
// CHECK:STDOUT:   │ │ ├─IfStatementElse 'else'
// CHECK:STDOUT:   │ │ │   ╭─IfConditionStart '('
// CHECK:STDOUT:   │ │ │   ├─IdentifierNameExpr 'x'
// CHECK:STDOUT:   │ │ │ ╭─IfCondition ')'
// CHECK:STDOUT:   │ │ │ │ ╭─CodeBlockStart '{'
// CHECK:STDOUT:   │ │ │ │ │ ╭─IdentifierNameExpr 'g'
// CHECK:STDOUT:   │ │ │ │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │ │ ├─CodeBlock '}'
// CHECK:STDOUT:   │ │ │ ├─IfStatementElse 'else'
// CHECK:STDOUT:   │ │ │ │ ╭─CodeBlockStart '{'
// CHECK:STDOUT:   │ │ │ │ │ ╭─IdentifierNameExpr 'h'
// CHECK:STDOUT:   │ │ │ │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │ │ ├─CodeBlock '}'
// CHECK:STDOUT:   │ │ ├─IfStatement 'if'
// CHECK:STDOUT:   │ ├─IfStatement 'if'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
