// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/member_access/compound.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/member_access/compound.carbon

fn F() {
  a.(b);
  a->(b);
}

// CHECK:STDOUT: - filename: compound.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameExpr 'a'
// CHECK:STDOUT:   │ │   │ ╭─ParenExprStart '('
// CHECK:STDOUT:   │ │   │ ├─IdentifierNameExpr 'b'
// CHECK:STDOUT:   │ │   ├─ParenExpr ')'
// CHECK:STDOUT:   │ │ ╭─MemberAccessExpr '.'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameExpr 'a'
// CHECK:STDOUT:   │ │   │ ╭─ParenExprStart '('
// CHECK:STDOUT:   │ │   │ ├─IdentifierNameExpr 'b'
// CHECK:STDOUT:   │ │   ├─ParenExpr ')'
// CHECK:STDOUT:   │ │ ╭─PointerMemberAccessExpr '->'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
