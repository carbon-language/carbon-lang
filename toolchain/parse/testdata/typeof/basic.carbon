// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/typeof/basic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/typeof/basic.carbon

// --- basic.carbon

var x: typeof(y);

// --- member_access.carbon

fn F() {
  typeof(x).G();
}

// --- fail_no_paren.carbon

// CHECK:STDERR: fail_no_paren.carbon:[[@LINE+4]]:15: error: expected `(` after `typeof` [ExpectedParenAfter]
// CHECK:STDERR: var x: typeof y;
// CHECK:STDERR:               ^
// CHECK:STDERR:
var x: typeof y;

// --- fail_no_close_paren.carbon

// CHECK:STDERR: fail_no_close_paren.carbon:[[@LINE+4]]:17: error: unexpected tokens before `)` [ExpectedCloseSymbol]
// CHECK:STDERR: var x: typeof(y z);
// CHECK:STDERR:                 ^
// CHECK:STDERR:
var x: typeof(y z);

// CHECK:STDOUT: - filename: basic.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   │ ╭─TypeOfExprKeyword 'typeof'
// CHECK:STDOUT:   │ │   │ ├─TypeOfExprOpenParen '('
// CHECK:STDOUT:   │ │   │ ├─IdentifierNameExpr 'y'
// CHECK:STDOUT:   │ │   ├─TypeOfExpr ')'
// CHECK:STDOUT:   │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ ├─VariablePattern 'var'
// CHECK:STDOUT:   ├─VariableDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: member_access.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │         ╭─TypeOfExprKeyword 'typeof'
// CHECK:STDOUT:   │ │         ├─TypeOfExprOpenParen '('
// CHECK:STDOUT:   │ │         ├─IdentifierNameExpr 'x'
// CHECK:STDOUT:   │ │       ╭─TypeOfExpr ')'
// CHECK:STDOUT:   │ │       ├─IdentifierNameNotBeforeSignature 'G'
// CHECK:STDOUT:   │ │     ╭─MemberAccessExpr '.'
// CHECK:STDOUT:   │ │   ╭─CallExprStart '('
// CHECK:STDOUT:   │ │ ╭─CallExpr ')'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_no_paren.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   │ ╭─TypeOfExprKeyword 'typeof'
// CHECK:STDOUT:   │ │   │ ├─TypeOfExprOpenParen 'typeof' has_error
// CHECK:STDOUT:   │ │   │ ├─IdentifierNameExpr 'y'
// CHECK:STDOUT:   │ │   ├─TypeOfExpr 'typeof' has_error
// CHECK:STDOUT:   │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ ├─VariablePattern 'var'
// CHECK:STDOUT:   ├─VariableDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_no_close_paren.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   │ ╭─TypeOfExprKeyword 'typeof'
// CHECK:STDOUT:   │ │   │ ├─TypeOfExprOpenParen '('
// CHECK:STDOUT:   │ │   │ ├─IdentifierNameExpr 'y'
// CHECK:STDOUT:   │ │   ├─TypeOfExpr ')' has_error
// CHECK:STDOUT:   │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ ├─VariablePattern 'var'
// CHECK:STDOUT:   ├─VariableDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
