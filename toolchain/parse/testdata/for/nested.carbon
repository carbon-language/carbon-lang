// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/for/nested.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/for/nested.carbon

fn foo() {
  for (var y: i32 in x) {
      for (var z: i32 in y) {
          Print(z);
      }
  }
}

// CHECK:STDOUT: - filename: nested.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'foo'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │   ╭─ForHeaderStart '('
// CHECK:STDOUT:   │ │   │     ╭─IdentifierNameNotBeforeSignature 'y'
// CHECK:STDOUT:   │ │   │     ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   │     ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │   │   ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ │   │ ╭─VariablePattern 'var'
// CHECK:STDOUT:   │ │   ├─ForIn 'in'
// CHECK:STDOUT:   │ │   ├─IdentifierNameExpr 'x'
// CHECK:STDOUT:   │ │ ╭─ForHeader ')'
// CHECK:STDOUT:   │ │ │ ╭─CodeBlockStart '{'
// CHECK:STDOUT:   │ │ │ │   ╭─ForHeaderStart '('
// CHECK:STDOUT:   │ │ │ │   │     ╭─IdentifierNameNotBeforeSignature 'z'
// CHECK:STDOUT:   │ │ │ │   │     ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ │   │     ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ │ │   │   ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ │ │ │   │ ╭─VariablePattern 'var'
// CHECK:STDOUT:   │ │ │ │   ├─ForIn 'in'
// CHECK:STDOUT:   │ │ │ │   ├─IdentifierNameExpr 'y'
// CHECK:STDOUT:   │ │ │ │ ╭─ForHeader ')'
// CHECK:STDOUT:   │ │ │ │ │ ╭─CodeBlockStart '{'
// CHECK:STDOUT:   │ │ │ │ │ │     ╭─IdentifierNameExpr 'Print'
// CHECK:STDOUT:   │ │ │ │ │ │   ╭─CallExprStart '('
// CHECK:STDOUT:   │ │ │ │ │ │   ├─IdentifierNameExpr 'z'
// CHECK:STDOUT:   │ │ │ │ │ │ ╭─CallExpr ')'
// CHECK:STDOUT:   │ │ │ │ │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │ │ │ ├─CodeBlock '}'
// CHECK:STDOUT:   │ │ │ ├─ForStatement 'for'
// CHECK:STDOUT:   │ │ ├─CodeBlock '}'
// CHECK:STDOUT:   │ ├─ForStatement 'for'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
