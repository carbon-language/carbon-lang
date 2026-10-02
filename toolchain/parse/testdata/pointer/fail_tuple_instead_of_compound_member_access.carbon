// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/pointer/fail_tuple_instead_of_compound_member_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/pointer/fail_tuple_instead_of_compound_member_access.carbon

fn F() {
  // CHECK:STDERR: fail_tuple_instead_of_compound_member_access.carbon:[[@LINE+2]]:6: error: expected expression [ExpectedExpr]
  // CHECK:STDERR:
  x.();

  // CHECK:STDERR: fail_tuple_instead_of_compound_member_access.carbon:[[@LINE+2]]:7: error: expected `)` [UnexpectedTokenInCompoundMemberAccess]
  // CHECK:STDERR:
  y.(1,);

  // CHECK:STDERR: fail_tuple_instead_of_compound_member_access.carbon:[[@LINE+2]]:7: error: expected `)` [UnexpectedTokenInCompoundMemberAccess]
  // CHECK:STDERR:
  z.(a, b);

  // CHECK:STDERR: fail_tuple_instead_of_compound_member_access.carbon:[[@LINE+2]]:8: error: expected `)` [UnexpectedTokenInCompoundMemberAccess]
  // CHECK:STDERR:
  w.(c d e ());

  // CHECK:STDERR: fail_tuple_instead_of_compound_member_access.carbon:[[@LINE+2]]:7: error: expected expression [ExpectedExpr]
  // CHECK:STDERR:
  v->();

  // CHECK:STDERR: fail_tuple_instead_of_compound_member_access.carbon:[[@LINE+2]]:8: error: expected `)` [UnexpectedTokenInCompoundMemberAccess]
  // CHECK:STDERR:
  u->(f,);

  // CHECK:STDERR: fail_tuple_instead_of_compound_member_access.carbon:[[@LINE+2]]:8: error: expected `)` [UnexpectedTokenInCompoundMemberAccess]
  // CHECK:STDERR:
  t->(2, 3);

  // CHECK:STDERR: fail_tuple_instead_of_compound_member_access.carbon:[[@LINE+2]]:9: error: expected `)` [UnexpectedTokenInCompoundMemberAccess]
  // CHECK:STDERR:
  s->(g h i ());
}

// CHECK:STDOUT: - filename: fail_tuple_instead_of_compound_member_access.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameExpr 'x'
// CHECK:STDOUT:   │ │   │ ╭─ParenExprStart '('
// CHECK:STDOUT:   │ │   │ ├─InvalidParse ')' has_error
// CHECK:STDOUT:   │ │   ├─ParenExpr ')' has_error
// CHECK:STDOUT:   │ │ ╭─MemberAccessExpr '.'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameExpr 'y'
// CHECK:STDOUT:   │ │   │ ╭─ParenExprStart '('
// CHECK:STDOUT:   │ │   │ ├─IntLiteral '1'
// CHECK:STDOUT:   │ │   ├─ParenExpr ')' has_error
// CHECK:STDOUT:   │ │ ╭─MemberAccessExpr '.'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameExpr 'z'
// CHECK:STDOUT:   │ │   │ ╭─ParenExprStart '('
// CHECK:STDOUT:   │ │   │ ├─IdentifierNameExpr 'a'
// CHECK:STDOUT:   │ │   ├─ParenExpr ')' has_error
// CHECK:STDOUT:   │ │ ╭─MemberAccessExpr '.'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameExpr 'w'
// CHECK:STDOUT:   │ │   │ ╭─ParenExprStart '('
// CHECK:STDOUT:   │ │   │ ├─IdentifierNameExpr 'c'
// CHECK:STDOUT:   │ │   ├─ParenExpr ')' has_error
// CHECK:STDOUT:   │ │ ╭─MemberAccessExpr '.'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameExpr 'v'
// CHECK:STDOUT:   │ │   │ ╭─ParenExprStart '('
// CHECK:STDOUT:   │ │   │ ├─InvalidParse ')' has_error
// CHECK:STDOUT:   │ │   ├─ParenExpr ')' has_error
// CHECK:STDOUT:   │ │ ╭─PointerMemberAccessExpr '->'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameExpr 'u'
// CHECK:STDOUT:   │ │   │ ╭─ParenExprStart '('
// CHECK:STDOUT:   │ │   │ ├─IdentifierNameExpr 'f'
// CHECK:STDOUT:   │ │   ├─ParenExpr ')' has_error
// CHECK:STDOUT:   │ │ ╭─PointerMemberAccessExpr '->'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameExpr 't'
// CHECK:STDOUT:   │ │   │ ╭─ParenExprStart '('
// CHECK:STDOUT:   │ │   │ ├─IntLiteral '2'
// CHECK:STDOUT:   │ │   ├─ParenExpr ')' has_error
// CHECK:STDOUT:   │ │ ╭─PointerMemberAccessExpr '->'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameExpr 's'
// CHECK:STDOUT:   │ │   │ ╭─ParenExprStart '('
// CHECK:STDOUT:   │ │   │ ├─IdentifierNameExpr 'g'
// CHECK:STDOUT:   │ │   ├─ParenExpr ')' has_error
// CHECK:STDOUT:   │ │ ╭─PointerMemberAccessExpr '->'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
