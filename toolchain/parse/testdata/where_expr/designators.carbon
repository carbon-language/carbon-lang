// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/where_expr/designators.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/where_expr/designators.carbon

// --- designator.carbon

fn Success() {
  .x;
  .Self;
}

// --- fail_numeric_designator.carbon

fn FailNumeric() {
  // CHECK:STDERR: fail_numeric_designator.carbon:[[@LINE+2]]:4: error: expected identifier or `Self` after `.` [ExpectedIdentifierOrSelfAfterPeriod]
  // CHECK:STDERR:
  .1;
}

// --- fail_keyword_designator.carbon

fn FailKeyword() {
  // CHECK:STDERR: fail_keyword_designator.carbon:[[@LINE+2]]:4-7: error: expected identifier or `Self` after `.` [ExpectedIdentifierOrSelfAfterPeriod]
  // CHECK:STDERR:
  .base;
}

// --- fail_empty_designator.carbon

fn FailEmpty() {
  // CHECK:STDERR: fail_empty_designator.carbon:[[@LINE+2]]:4: error: expected identifier or `Self` after `.` [ExpectedIdentifierOrSelfAfterPeriod]
  // CHECK:STDERR:
  .;
}

// --- fail_star_designator.carbon

fn FailStar() {
  // CHECK:STDERR: fail_star_designator.carbon:[[@LINE+2]]:4: error: expected identifier or `Self` after `.` [ExpectedIdentifierOrSelfAfterPeriod]
  // CHECK:STDERR:
  .*;
}

// --- fail_designator_eof.carbon

var x: i32 = .

// CHECK:STDERR: fail_designator_eof.carbon:[[@LINE+86]]:26: error: expected identifier or `Self` after `.` [ExpectedIdentifierOrSelfAfterPeriod]
// CHECK:STDERR:
// CHECK:STDERR: fail_designator_eof.carbon:[[@LINE+84]]:26: error: `var` declarations must end with a `;` [ExpectedDeclSemi]
// CHECK:STDERR:
// CHECK:STDOUT: - filename: designator.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'Success'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ ╭─DesignatorExpr '.'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │   ╭─SelfTypeName 'Self'
// CHECK:STDOUT:   │ │ ╭─DesignatorExpr '.'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_numeric_designator.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'FailNumeric'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │   ╭─InvalidParse '1' has_error
// CHECK:STDOUT:   │ │ ╭─DesignatorExpr '.' has_error
// CHECK:STDOUT:   │ ├─ExprStatement ';' has_error
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_keyword_designator.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'FailKeyword'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameNotBeforeSignature 'base' has_error
// CHECK:STDOUT:   │ │ ╭─DesignatorExpr '.' has_error
// CHECK:STDOUT:   │ ├─ExprStatement ';' has_error
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_empty_designator.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'FailEmpty'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │   ╭─InvalidParse ';' has_error
// CHECK:STDOUT:   │ │ ╭─DesignatorExpr '.' has_error
// CHECK:STDOUT:   │ ├─ExprStatement ';' has_error
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_star_designator.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'FailStar'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │     ╭─InvalidParse '*' has_error
// CHECK:STDOUT:   │ │   ╭─DesignatorExpr '.' has_error
// CHECK:STDOUT:   │ │ ╭─PostfixOperatorStar '*' has_error
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_designator_eof.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ ├─VariableInitializer '='
// CHECK:STDOUT:   │ │ ╭─InvalidParse '' has_error
// CHECK:STDOUT:   │ ├─DesignatorExpr '.' has_error
// CHECK:STDOUT:   ├─VariableDecl '.' has_error
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
