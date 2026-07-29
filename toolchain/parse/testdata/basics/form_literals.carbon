// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/basics/form_literals.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/basics/form_literals.carbon

// --- basics.carbon

fn F() {
  form(ref i32);
  form(var T(X, Y));
  form(val X.Y);
}

fn VarReturn() ->? form(ref i32);

// --- fail_invalid_form_literal.carbon

fn F() {
  // CHECK:STDERR: fail_invalid_form_literal.carbon:[[@LINE+4]]:7: error: expected `(` after `form` [ExpectedParenAfter]
  // CHECK:STDERR:   form;
  // CHECK:STDERR:       ^
  // CHECK:STDERR:
  form;
  // CHECK:STDERR: fail_invalid_form_literal.carbon:[[@LINE+8]]:8: error: expected `ref`, `var`, or `val` after `form(` [ExpectedCategoryModifier]
  // CHECK:STDERR:   form();
  // CHECK:STDERR:        ^
  // CHECK:STDERR:
  // CHECK:STDERR: fail_invalid_form_literal.carbon:[[@LINE+4]]:8: error: expected expression [ExpectedExpr]
  // CHECK:STDERR:   form();
  // CHECK:STDERR:        ^
  // CHECK:STDERR:
  form();
  // CHECK:STDERR: fail_invalid_form_literal.carbon:[[@LINE+4]]:8: error: expected `ref`, `var`, or `val` after `form(` [ExpectedCategoryModifier]
  // CHECK:STDERR:   form(i32);
  // CHECK:STDERR:        ^~~
  // CHECK:STDERR:
  form(i32);
  // CHECK:STDERR: fail_invalid_form_literal.carbon:[[@LINE+8]]:8: error: expected `ref`, `var`, or `val` after `form(` [ExpectedCategoryModifier]
  // CHECK:STDERR:   form(whole lot of nonsense);
  // CHECK:STDERR:        ^~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_invalid_form_literal.carbon:[[@LINE+4]]:14: error: unexpected tokens before `)` [ExpectedCloseSymbol]
  // CHECK:STDERR:   form(whole lot of nonsense);
  // CHECK:STDERR:              ^~~
  // CHECK:STDERR:
  form(whole lot of nonsense);
  // CHECK:STDERR: fail_invalid_form_literal.carbon:[[@LINE+4]]:18: error: unexpected tokens before `)` [ExpectedCloseSymbol]
  // CHECK:STDERR:   form(var whole lot of nonsense);
  // CHECK:STDERR:                  ^~~
  // CHECK:STDERR:
  form(var whole lot of nonsense);
  // CHECK:STDERR: fail_invalid_form_literal.carbon:[[@LINE+8]]:8: error: expected `(` after `form` [ExpectedParenAfter]
  // CHECK:STDERR:   form whole lot of nonsense;
  // CHECK:STDERR:        ^~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_invalid_form_literal.carbon:[[@LINE+4]]:8: error: expected `;` after expression statement [ExpectedExprSemi]
  // CHECK:STDERR:   form whole lot of nonsense;
  // CHECK:STDERR:        ^~~~~
  // CHECK:STDERR:
  form whole lot of nonsense;
  // CHECK:STDERR: fail_invalid_form_literal.carbon:[[@LINE+8]]:8: error: expected `(` after `form` [ExpectedParenAfter]
  // CHECK:STDERR:   form var whole lot of nonsense;
  // CHECK:STDERR:        ^~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_invalid_form_literal.carbon:[[@LINE+4]]:8: error: expected `;` after expression statement [ExpectedExprSemi]
  // CHECK:STDERR:   form var whole lot of nonsense;
  // CHECK:STDERR:        ^~~
  // CHECK:STDERR:
  form var whole lot of nonsense;
}

// CHECK:STDOUT: - filename: basics.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │   ╭─FormLiteralKeyword 'form'
// CHECK:STDOUT:   │ │   ├─FormLiteralOpenParen '('
// CHECK:STDOUT:   │ │   │ ╭─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │   ├─RefPrimitiveForm 'ref'
// CHECK:STDOUT:   │ │ ╭─FormLiteral ')'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │   ╭─FormLiteralKeyword 'form'
// CHECK:STDOUT:   │ │   ├─FormLiteralOpenParen '('
// CHECK:STDOUT:   │ │   │     ╭─IdentifierNameExpr 'T'
// CHECK:STDOUT:   │ │   │   ╭─CallExprStart '('
// CHECK:STDOUT:   │ │   │   ├─IdentifierNameExpr 'X'
// CHECK:STDOUT:   │ │   │   ├─TupleLiteralComma ','
// CHECK:STDOUT:   │ │   │   ├─IdentifierNameExpr 'Y'
// CHECK:STDOUT:   │ │   │ ╭─CallExpr ')'
// CHECK:STDOUT:   │ │   ├─VarPrimitiveForm 'var'
// CHECK:STDOUT:   │ │ ╭─FormLiteral ')'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │   ╭─FormLiteralKeyword 'form'
// CHECK:STDOUT:   │ │   ├─FormLiteralOpenParen '('
// CHECK:STDOUT:   │ │   │   ╭─IdentifierNameExpr 'X'
// CHECK:STDOUT:   │ │   │   ├─IdentifierNameNotBeforeSignature 'Y'
// CHECK:STDOUT:   │ │   │ ╭─MemberAccessExpr '.'
// CHECK:STDOUT:   │ │   ├─ValPrimitiveForm 'val'
// CHECK:STDOUT:   │ │ ╭─FormLiteral ')'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'VarReturn'
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ │   ╭─FormLiteralKeyword 'form'
// CHECK:STDOUT:   │ │   ├─FormLiteralOpenParen '('
// CHECK:STDOUT:   │ │   │ ╭─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │   ├─RefPrimitiveForm 'ref'
// CHECK:STDOUT:   │ │ ╭─FormLiteral ')'
// CHECK:STDOUT:   │ ├─ReturnForm '->?'
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_invalid_form_literal.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │   ╭─FormLiteralKeyword 'form'
// CHECK:STDOUT:   │ │   ├─FormLiteralOpenParen 'form' ❌
// CHECK:STDOUT:   │ │   ├─InvalidParse 'form' ❌
// CHECK:STDOUT:   │ │ ╭─FormLiteral 'form' ❌
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │   ╭─FormLiteralKeyword 'form'
// CHECK:STDOUT:   │ │   ├─FormLiteralOpenParen '('
// CHECK:STDOUT:   │ │   │ ╭─InvalidParse ')' ❌
// CHECK:STDOUT:   │ │   ├─ValPrimitiveForm 'form' ❌
// CHECK:STDOUT:   │ │ ╭─FormLiteral ')'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │   ╭─FormLiteralKeyword 'form'
// CHECK:STDOUT:   │ │   ├─FormLiteralOpenParen '('
// CHECK:STDOUT:   │ │   │ ╭─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │   ├─ValPrimitiveForm 'form' ❌
// CHECK:STDOUT:   │ │ ╭─FormLiteral ')'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │   ╭─FormLiteralKeyword 'form'
// CHECK:STDOUT:   │ │   ├─FormLiteralOpenParen '('
// CHECK:STDOUT:   │ │   │ ╭─IdentifierNameExpr 'whole'
// CHECK:STDOUT:   │ │   ├─ValPrimitiveForm 'form' ❌
// CHECK:STDOUT:   │ │ ╭─FormLiteral ')' ❌
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │   ╭─FormLiteralKeyword 'form'
// CHECK:STDOUT:   │ │   ├─FormLiteralOpenParen '('
// CHECK:STDOUT:   │ │   │ ╭─IdentifierNameExpr 'whole'
// CHECK:STDOUT:   │ │   ├─VarPrimitiveForm 'var'
// CHECK:STDOUT:   │ │ ╭─FormLiteral ')' ❌
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │   ╭─FormLiteralKeyword 'form'
// CHECK:STDOUT:   │ │   ├─FormLiteralOpenParen 'form' ❌
// CHECK:STDOUT:   │ │   ├─InvalidParse 'form' ❌
// CHECK:STDOUT:   │ │ ╭─FormLiteral 'form' ❌
// CHECK:STDOUT:   │ ├─ExprStatement ';' ❌
// CHECK:STDOUT:   │ │   ╭─FormLiteralKeyword 'form'
// CHECK:STDOUT:   │ │   ├─FormLiteralOpenParen 'form' ❌
// CHECK:STDOUT:   │ │   ├─InvalidParse 'form' ❌
// CHECK:STDOUT:   │ │ ╭─FormLiteral 'form' ❌
// CHECK:STDOUT:   │ ├─ExprStatement ';' ❌
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
