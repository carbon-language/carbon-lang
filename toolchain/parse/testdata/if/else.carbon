// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/if/else.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/if/else.carbon

fn F() {
  if (a) {
    if (b) {
      c;
    } else {
      d;
    }
  } else {
    e;
  }
  if (x) { f; }
  else if (x) { g; }
  else { h; }
}

// CHECK:STDOUT: - filename: else.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │   ╭─IfConditionStart '('
// CHECK:STDOUT:   │ │   ├─IdentifierNameExpr 'a'
// CHECK:STDOUT:   │ │ ╭─IfCondition ')'
// CHECK:STDOUT:   │ │ │ ╭─CodeBlockStart '{'
// CHECK:STDOUT:   │ │ │ │   ╭─IfConditionStart '('
// CHECK:STDOUT:   │ │ │ │   ├─IdentifierNameExpr 'b'
// CHECK:STDOUT:   │ │ │ │ ╭─IfCondition ')'
// CHECK:STDOUT:   │ │ │ │ │ ╭─CodeBlockStart '{'
// CHECK:STDOUT:   │ │ │ │ │ │ ╭─IdentifierNameExpr 'c'
// CHECK:STDOUT:   │ │ │ │ │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │ │ │ ├─CodeBlock '}'
// CHECK:STDOUT:   │ │ │ │ ├─IfStatementElse 'else'
// CHECK:STDOUT:   │ │ │ │ │ ╭─CodeBlockStart '{'
// CHECK:STDOUT:   │ │ │ │ │ │ ╭─IdentifierNameExpr 'd'
// CHECK:STDOUT:   │ │ │ │ │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │ │ │ ├─CodeBlock '}'
// CHECK:STDOUT:   │ │ │ ├─IfStatement 'if'
// CHECK:STDOUT:   │ │ ├─CodeBlock '}'
// CHECK:STDOUT:   │ │ ├─IfStatementElse 'else'
// CHECK:STDOUT:   │ │ │ ╭─CodeBlockStart '{'
// CHECK:STDOUT:   │ │ │ │ ╭─IdentifierNameExpr 'e'
// CHECK:STDOUT:   │ │ │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │ ├─CodeBlock '}'
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
