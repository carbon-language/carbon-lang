// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/member_access/simple.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/member_access/simple.carbon

fn F(a: T) {
  a.x;
  a->x;
  a.x->y;
  a->x.y;
}

// CHECK:STDOUT: - filename: simple.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   │ │ ╭─IdentifierNameNotBeforeSignature 'a'
// CHECK:STDOUT:   │   │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │   │ │ ├─IdentifierNameExpr 'T'
// CHECK:STDOUT:   │   │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameExpr 'a'
// CHECK:STDOUT:   │ │   ├─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ ╭─MemberAccessExpr '.'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameExpr 'a'
// CHECK:STDOUT:   │ │   ├─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ ╭─PointerMemberAccessExpr '->'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │     ╭─IdentifierNameExpr 'a'
// CHECK:STDOUT:   │ │     ├─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │   ╭─MemberAccessExpr '.'
// CHECK:STDOUT:   │ │   ├─IdentifierNameNotBeforeSignature 'y'
// CHECK:STDOUT:   │ │ ╭─PointerMemberAccessExpr '->'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │     ╭─IdentifierNameExpr 'a'
// CHECK:STDOUT:   │ │     ├─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │   ╭─PointerMemberAccessExpr '->'
// CHECK:STDOUT:   │ │   ├─IdentifierNameNotBeforeSignature 'y'
// CHECK:STDOUT:   │ │ ╭─MemberAccessExpr '.'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
