// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/basics/fail_missing_token_locations.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/basics/fail_missing_token_locations.carbon

// --- fail_semi_before_comment.carbon

// The `;` goes after the `1`, before the spaces and the comment.
// CHECK:STDERR: fail_semi_before_comment.carbon:[[@LINE+2]]:15: error: `var` declarations must end with a `;` [ExpectedDeclSemi]
// CHECK:STDERR:
var x: i32 = 1  // The value.
var y: i32 = 2;

// --- fail_expr_at_line_end.carbon

fn F() {
  // CHECK:STDERR: fail_expr_at_line_end.carbon:[[@LINE+4]]:15: error: expected expression [ExpectedExpr]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_expr_at_line_end.carbon:[[@LINE+2]]:15: error: `var` declarations must end with a `;` [ExpectedDeclSemi]
  // CHECK:STDERR:
  var x: i32 =
  var y: i32 = 2;
}

// --- fail_expr_at_statement_start.carbon

// The statement starts at the `=`, so the missing expression goes there rather
// than at the end of the line before.
fn F() {
  var x: i32 = 1;
  // CHECK:STDERR: fail_expr_at_statement_start.carbon:[[@LINE+2]]:3: error: expected expression [ExpectedExpr]
  // CHECK:STDERR:
  = 3;
}

// CHECK:STDOUT: - filename: fail_semi_before_comment.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ ├─VariableInitializer '='
// CHECK:STDOUT:   │ ├─IntLiteral '1'
// CHECK:STDOUT:   ├─VariableDecl ';' has_error
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_expr_at_line_end.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │   ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ │ ├─VariableInitializer '='
// CHECK:STDOUT:   │ │ ├─InvalidParse 'var' has_error
// CHECK:STDOUT:   │ ├─VariableDecl ';' has_error
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_expr_at_statement_start.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │   ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ │ ├─VariableInitializer '='
// CHECK:STDOUT:   │ │ ├─IntLiteral '1'
// CHECK:STDOUT:   │ ├─VariableDecl ';'
// CHECK:STDOUT:   │ │   ╭─InvalidParse '=' has_error
// CHECK:STDOUT:   │ │   ├─IntLiteral '3'
// CHECK:STDOUT:   │ │ ╭─InfixOperatorEqual '=' has_error
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
