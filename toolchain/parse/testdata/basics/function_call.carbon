// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/basics/function_call.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/basics/function_call.carbon

// NOTE: Move to its own directory when more tests are added.
fn F() {
  a.b.f(c.d, (e)).g();
}

// CHECK:STDOUT: - filename: function_call.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │               ╭─IdentifierNameExpr 'a'
// CHECK:STDOUT:   │ │               ├─IdentifierNameNotBeforeSignature 'b'
// CHECK:STDOUT:   │ │             ╭─MemberAccessExpr '.'
// CHECK:STDOUT:   │ │             ├─IdentifierNameNotBeforeSignature 'f'
// CHECK:STDOUT:   │ │           ╭─MemberAccessExpr '.'
// CHECK:STDOUT:   │ │         ╭─CallExprStart '('
// CHECK:STDOUT:   │ │         │ ╭─IdentifierNameExpr 'c'
// CHECK:STDOUT:   │ │         │ ├─IdentifierNameNotBeforeSignature 'd'
// CHECK:STDOUT:   │ │         ├─MemberAccessExpr '.'
// CHECK:STDOUT:   │ │         ├─TupleLiteralComma ','
// CHECK:STDOUT:   │ │         │ ╭─ParenExprStart '('
// CHECK:STDOUT:   │ │         │ ├─IdentifierNameExpr 'e'
// CHECK:STDOUT:   │ │         ├─ParenExpr ')'
// CHECK:STDOUT:   │ │       ╭─CallExpr ')'
// CHECK:STDOUT:   │ │       ├─IdentifierNameNotBeforeSignature 'g'
// CHECK:STDOUT:   │ │     ╭─MemberAccessExpr '.'
// CHECK:STDOUT:   │ │   ╭─CallExprStart '('
// CHECK:STDOUT:   │ │ ╭─CallExpr ')'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
