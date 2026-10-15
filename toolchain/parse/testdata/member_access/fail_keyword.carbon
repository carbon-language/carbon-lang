// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/member_access/fail_keyword.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/member_access/fail_keyword.carbon

fn F() {
  // CHECK:STDERR: fail_keyword.carbon:[[@LINE+2]]:5-8: error: expected identifier after `.`; prefix reserved word with `r#` to form a valid identifier [ExpectedIdentifierAfterPeriodOrArrow]
  // CHECK:STDERR:
  a.self;
  // CHECK:STDERR: fail_keyword.carbon:[[@LINE+2]]:5-8: error: expected identifier after `.`; prefix reserved word with `r#` to form a valid identifier [ExpectedIdentifierAfterPeriodOrArrow]
  // CHECK:STDERR:
  a.Self;
  // CHECK:STDERR: fail_keyword.carbon:[[@LINE+2]]:5-7: error: expected identifier after `.`; prefix reserved word with `r#` to form a valid identifier [ExpectedIdentifierAfterPeriodOrArrow]
  // CHECK:STDERR:
  a.for;
  // CHECK:STDERR: fail_keyword.carbon:[[@LINE+2]]:6-9: error: expected identifier after `->`; prefix reserved word with `r#` to form a valid identifier [ExpectedIdentifierAfterPeriodOrArrow]
  // CHECK:STDERR:
  p->self;
  // CHECK:STDERR: fail_keyword.carbon:[[@LINE+2]]:6-9: error: expected identifier after `->`; prefix reserved word with `r#` to form a valid identifier [ExpectedIdentifierAfterPeriodOrArrow]
  // CHECK:STDERR:
  p->Self;
  // CHECK:STDERR: fail_keyword.carbon:[[@LINE+2]]:6-10: error: expected identifier after `->`; prefix reserved word with `r#` to form a valid identifier [ExpectedIdentifierAfterPeriodOrArrow]
  // CHECK:STDERR:
  p->while;
}

// CHECK:STDOUT: - filename: fail_keyword.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameExpr 'a'
// CHECK:STDOUT:   │ │   ├─IdentifierNameNotBeforeSignature 'self'
// CHECK:STDOUT:   │ │ ╭─MemberAccessExpr '.'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameExpr 'a'
// CHECK:STDOUT:   │ │   ├─IdentifierNameNotBeforeSignature 'Self'
// CHECK:STDOUT:   │ │ ╭─MemberAccessExpr '.'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameExpr 'a'
// CHECK:STDOUT:   │ │   ├─IdentifierNameNotBeforeSignature 'for'
// CHECK:STDOUT:   │ │ ╭─MemberAccessExpr '.'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameExpr 'p'
// CHECK:STDOUT:   │ │   ├─IdentifierNameNotBeforeSignature 'self'
// CHECK:STDOUT:   │ │ ╭─PointerMemberAccessExpr '->'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameExpr 'p'
// CHECK:STDOUT:   │ │   ├─IdentifierNameNotBeforeSignature 'Self'
// CHECK:STDOUT:   │ │ ╭─PointerMemberAccessExpr '->'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameExpr 'p'
// CHECK:STDOUT:   │ │   ├─IdentifierNameNotBeforeSignature 'while'
// CHECK:STDOUT:   │ │ ╭─PointerMemberAccessExpr '->'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
