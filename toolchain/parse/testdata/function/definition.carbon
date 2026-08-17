// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/function/definition.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/function/definition.carbon

// --- basic.carbon

fn F() {}

// --- with_params.carbon

fn foo(bar: i64, baz: i64) {
  foo(baz, bar + baz);
}

// --- with_positional_params.carbon

fn foo {
  foo($0, $42 + $86);
}

// --- fail_positional_param_negative_int.carbon

fn foo {
  // CHECK:STDERR: fail_positional_param_negative_int.carbon:[[@LINE+8]]:8: error: expected unsigned integer literal after `$` [ExpectedUnsignedIntLiteralAfterDollar]
  // CHECK:STDERR:   foo($-1);
  // CHECK:STDERR:        ^
  // CHECK:STDERR:
  // CHECK:STDERR: fail_positional_param_negative_int.carbon:[[@LINE+4]]:8: error: whitespace missing around binary operator [BinaryOperatorRequiresWhitespace]
  // CHECK:STDERR:   foo($-1);
  // CHECK:STDERR:        ^
  // CHECK:STDERR:
  foo($-1);
}

// --- fail_positional_param_float.carbon

fn foo {
  // CHECK:STDERR: fail_positional_param_float.carbon:[[@LINE+4]]:8: error: expected unsigned integer literal after `$` [ExpectedUnsignedIntLiteralAfterDollar]
  // CHECK:STDERR:   foo($1.0);
  // CHECK:STDERR:        ^~~
  // CHECK:STDERR:
  foo($1.0);
}

// --- with_return_type.carbon

fn foo() -> f64 {
  return 42;
}

// --- fail_identifier_in_statements.carbon

fn F() {
  // Note: this might become valid depending on the expression syntax. This test
  // shouldn't be taken as a sign it should remain invalid.
  bar
// CHECK:STDERR: fail_identifier_in_statements.carbon:[[@LINE+4]]:1: error: expected `;` after expression statement [ExpectedExprSemi]
// CHECK:STDERR: }
// CHECK:STDERR: ^
// CHECK:STDERR:
}

// --- nested.carbon

fn F() {
  fn NamedParams(a: i32) {}
  fn PositionalParams {}
}

// --- builtin.carbon

fn F() = "builtin";

// --- builtin_nested.carbon

impl T as I {
  fn Op() = "builtin";
}

// --- fail_builtin_identifier.carbon

// CHECK:STDERR: fail_builtin_identifier.carbon:[[@LINE+4]]:18: error: expected builtin function name after `=` [ExpectedBuiltinName]
// CHECK:STDERR: fn NotString() = banana;
// CHECK:STDERR:                  ^~~~~~
// CHECK:STDERR:
fn NotString() = banana;

// --- fail_builtin_two_strings.carbon

// CHECK:STDERR: fail_builtin_two_strings.carbon:[[@LINE+4]]:32: error: `fn` declarations must end with a `;` [ExpectedDeclSemi]
// CHECK:STDERR: fn JunkAfterString() = "hello" "world";
// CHECK:STDERR:                                ^~~~~~~
// CHECK:STDERR:
fn JunkAfterString() = "hello" "world";

// --- fail_builtin_spurious_equals.carbon

// CHECK:STDERR: fail_builtin_spurious_equals.carbon:[[@LINE+4]]:23: error: expected builtin function name after `=` [ExpectedBuiltinName]
// CHECK:STDERR: fn SpuriousEquals() = { }
// CHECK:STDERR:                       ^
// CHECK:STDERR:
fn SpuriousEquals() = { }
fn TestRecoveryFromSpuriousEquals();

// CHECK:STDOUT: - filename: basic.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: with_params.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'foo'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   │ │ ╭─IdentifierNameNotBeforeSignature 'bar'
// CHECK:STDOUT:   │   │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │   │ │ ├─IntTypeLiteral 'i64'
// CHECK:STDOUT:   │   │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │   │ ├─PatternListComma ','
// CHECK:STDOUT:   │   │ │ ╭─IdentifierNameNotBeforeSignature 'baz'
// CHECK:STDOUT:   │   │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │   │ │ ├─IntTypeLiteral 'i64'
// CHECK:STDOUT:   │   │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │     ╭─IdentifierNameExpr 'foo'
// CHECK:STDOUT:   │ │   ╭─CallExprStart '('
// CHECK:STDOUT:   │ │   ├─IdentifierNameExpr 'baz'
// CHECK:STDOUT:   │ │   ├─TupleLiteralComma ','
// CHECK:STDOUT:   │ │   │ ╭─IdentifierNameExpr 'bar'
// CHECK:STDOUT:   │ │   │ ├─IdentifierNameExpr 'baz'
// CHECK:STDOUT:   │ │   ├─InfixOperatorPlus '+'
// CHECK:STDOUT:   │ │ ╭─CallExpr ')'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: with_positional_params.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'foo'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │     ╭─IdentifierNameExpr 'foo'
// CHECK:STDOUT:   │ │   ╭─CallExprStart '('
// CHECK:STDOUT:   │ │   │ ╭─IntLiteral '0'
// CHECK:STDOUT:   │ │   ├─PositionalParamExpr '$'
// CHECK:STDOUT:   │ │   ├─TupleLiteralComma ','
// CHECK:STDOUT:   │ │   │   ╭─IntLiteral '42'
// CHECK:STDOUT:   │ │   │ ╭─PositionalParamExpr '$'
// CHECK:STDOUT:   │ │   │ │ ╭─IntLiteral '86'
// CHECK:STDOUT:   │ │   │ ├─PositionalParamExpr '$'
// CHECK:STDOUT:   │ │   ├─InfixOperatorPlus '+'
// CHECK:STDOUT:   │ │ ╭─CallExpr ')'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_positional_param_negative_int.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'foo'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │     ╭─IdentifierNameExpr 'foo'
// CHECK:STDOUT:   │ │   ╭─CallExprStart '('
// CHECK:STDOUT:   │ │   │   ╭─InvalidParse '-' has_error
// CHECK:STDOUT:   │ │   │ ╭─PositionalParamExpr '$' has_error
// CHECK:STDOUT:   │ │   │ ├─IntLiteral '1'
// CHECK:STDOUT:   │ │   ├─InfixOperatorMinus '-' has_error
// CHECK:STDOUT:   │ │ ╭─CallExpr ')'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_positional_param_float.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'foo'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │     ╭─IdentifierNameExpr 'foo'
// CHECK:STDOUT:   │ │   ╭─CallExprStart '('
// CHECK:STDOUT:   │ │   │ ╭─InvalidParse '1.0' has_error
// CHECK:STDOUT:   │ │   ├─PositionalParamExpr '$' has_error
// CHECK:STDOUT:   │ │ ╭─CallExpr ')' has_error
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: with_return_type.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'foo'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │   │ ╭─FloatTypeLiteral 'f64'
// CHECK:STDOUT:   │   ├─ReturnType '->'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─ReturnStatementStart 'return'
// CHECK:STDOUT:   │ │ ├─IntLiteral '42'
// CHECK:STDOUT:   │ ├─ReturnStatement ';'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_identifier_in_statements.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─IdentifierNameExpr 'bar'
// CHECK:STDOUT:   │ ├─ExprStatement 'bar' has_error
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: nested.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ │   ├─IdentifierNameMaybeBeforeSignature 'NamedParams'
// CHECK:STDOUT:   │ │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │   │ │ ╭─IdentifierNameNotBeforeSignature 'a'
// CHECK:STDOUT:   │ │   │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │   │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ ├─FunctionDefinition '}'
// CHECK:STDOUT:   │ │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ │   ├─IdentifierNameNotBeforeSignature 'PositionalParams'
// CHECK:STDOUT:   │ │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: builtin.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─BuiltinFunctionDefinitionStart '='
// CHECK:STDOUT:   │ ├─BuiltinName '"builtin"'
// CHECK:STDOUT:   ├─BuiltinFunctionDefinition ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: builtin_nested.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─ImplIntroducer 'impl'
// CHECK:STDOUT:   │   │ ╭─IdentifierNameExpr 'T'
// CHECK:STDOUT:   │   ├─ImplTypeAs 'as'
// CHECK:STDOUT:   │   ├─IdentifierNameExpr 'I'
// CHECK:STDOUT:   │ ╭─ImplDefinitionStart '{'
// CHECK:STDOUT:   │ │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ │   ├─IdentifierNameMaybeBeforeSignature 'Op'
// CHECK:STDOUT:   │ │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ │ ╭─BuiltinFunctionDefinitionStart '='
// CHECK:STDOUT:   │ │ ├─BuiltinName '"builtin"'
// CHECK:STDOUT:   │ ├─BuiltinFunctionDefinition ';'
// CHECK:STDOUT:   ├─ImplDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_builtin_identifier.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'NotString'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─BuiltinFunctionDefinitionStart '='
// CHECK:STDOUT:   ├─BuiltinFunctionDefinition ';' has_error
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_builtin_two_strings.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'JunkAfterString'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─BuiltinFunctionDefinitionStart '='
// CHECK:STDOUT:   │ ├─BuiltinName '"hello"'
// CHECK:STDOUT:   ├─BuiltinFunctionDefinition ';' has_error
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_builtin_spurious_equals.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'SpuriousEquals'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─BuiltinFunctionDefinitionStart '='
// CHECK:STDOUT:   ├─BuiltinFunctionDefinition '}' has_error
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'TestRecoveryFromSpuriousEquals'
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
