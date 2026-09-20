// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/for/fail_colon_instead_of_in.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/for/fail_colon_instead_of_in.carbon

fn foo() {
  // CHECK:STDERR: fail_colon_instead_of_in.carbon:[[@LINE+2]]:19: error: `:` should be replaced by `in` [ExpectedInNotColon]
  // CHECK:STDERR:
  for (var x: i32 : y) {
    Print(x);
  }
}

// CHECK:STDOUT: - filename: fail_colon_instead_of_in.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'foo'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │   ╭─ForHeaderStart '('
// CHECK:STDOUT:   │ │   │     ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │   │     ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   │     ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │   │   ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ │   │ ╭─VariablePattern 'var'
// CHECK:STDOUT:   │ │   ├─ForIn ':' has_error
// CHECK:STDOUT:   │ │ ╭─ForHeader ')' has_error
// CHECK:STDOUT:   │ │ │ ╭─CodeBlockStart '{'
// CHECK:STDOUT:   │ │ │ │     ╭─IdentifierNameExpr 'Print'
// CHECK:STDOUT:   │ │ │ │   ╭─CallExprStart '('
// CHECK:STDOUT:   │ │ │ │   ├─IdentifierNameExpr 'x'
// CHECK:STDOUT:   │ │ │ │ ╭─CallExpr ')'
// CHECK:STDOUT:   │ │ │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │ ├─CodeBlock '}'
// CHECK:STDOUT:   │ ├─ForStatement 'for'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
