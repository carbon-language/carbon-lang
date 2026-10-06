// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/while/basic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/while/basic.carbon

fn F() {
  while (a) {
    if (b) {
      break;
    }
    if (c) {
      continue;
    }
  }
}

// CHECK:STDOUT: - filename: basic.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │   ╭─WhileConditionStart '('
// CHECK:STDOUT:   │ │   ├─IdentifierNameExpr 'a'
// CHECK:STDOUT:   │ │ ╭─WhileCondition ')'
// CHECK:STDOUT:   │ │ │ ╭─CodeBlockStart '{'
// CHECK:STDOUT:   │ │ │ │   ╭─IfConditionStart '('
// CHECK:STDOUT:   │ │ │ │   ├─IdentifierNameExpr 'b'
// CHECK:STDOUT:   │ │ │ │ ╭─IfCondition ')'
// CHECK:STDOUT:   │ │ │ │ │ ╭─CodeBlockStart '{'
// CHECK:STDOUT:   │ │ │ │ │ │ ╭─BreakStatementStart 'break'
// CHECK:STDOUT:   │ │ │ │ │ ├─BreakStatement ';'
// CHECK:STDOUT:   │ │ │ │ ├─CodeBlock '}'
// CHECK:STDOUT:   │ │ │ ├─IfStatement 'if'
// CHECK:STDOUT:   │ │ │ │   ╭─IfConditionStart '('
// CHECK:STDOUT:   │ │ │ │   ├─IdentifierNameExpr 'c'
// CHECK:STDOUT:   │ │ │ │ ╭─IfCondition ')'
// CHECK:STDOUT:   │ │ │ │ │ ╭─CodeBlockStart '{'
// CHECK:STDOUT:   │ │ │ │ │ │ ╭─ContinueStatementStart 'continue'
// CHECK:STDOUT:   │ │ │ │ │ ├─ContinueStatement ';'
// CHECK:STDOUT:   │ │ │ │ ├─CodeBlock '}'
// CHECK:STDOUT:   │ │ │ ├─IfStatement 'if'
// CHECK:STDOUT:   │ │ ├─CodeBlock '}'
// CHECK:STDOUT:   │ ├─WhileStatement 'while'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
