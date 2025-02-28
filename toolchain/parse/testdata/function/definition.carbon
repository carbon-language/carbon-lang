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
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'F'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 6},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: with_params.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'foo'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeParams', text: 'bar'},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i64'},
// CHECK:STDOUT:           {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeParams', text: 'baz'},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i64'},
// CHECK:STDOUT:           {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 9},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 12},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'foo'},
// CHECK:STDOUT:           {kind: 'CallExprStart', text: '(', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'baz'},
// CHECK:STDOUT:           {kind: 'CallExprComma', text: ','},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'bar'},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'baz'},
// CHECK:STDOUT:           {kind: 'InfixOperatorPlus', text: '+', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'CallExpr', text: ')', subtree_size: 8},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 9},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 22},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: with_return_type.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'foo'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'FloatTypeLiteral', text: 'f64'},
// CHECK:STDOUT:         {kind: 'ReturnType', text: '->', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 7},
// CHECK:STDOUT:         {kind: 'ReturnStatementStart', text: 'return'},
// CHECK:STDOUT:         {kind: 'IntLiteral', text: '42'},
// CHECK:STDOUT:       {kind: 'ReturnStatement', text: ';', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 11},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_identifier_in_statements.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'F'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'IdentifierNameExpr', text: 'bar'},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: 'bar', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 8},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: nested.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'F'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:           {kind: 'IdentifierNameBeforeParams', text: 'NamedParams'},
// CHECK:STDOUT:             {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeParams', text: 'a'},
// CHECK:STDOUT:               {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:             {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'ExplicitParamList', text: ')', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 8},
// CHECK:STDOUT:       {kind: 'FunctionDefinition', text: '}', subtree_size: 9},
// CHECK:STDOUT:           {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'PositionalParams'},
// CHECK:STDOUT:         {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'FunctionDefinition', text: '}', subtree_size: 4},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 19},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: builtin.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'F'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'BuiltinFunctionDefinitionStart', text: '=', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'BuiltinName', text: '"builtin"'},
// CHECK:STDOUT:     {kind: 'BuiltinFunctionDefinition', text: ';', subtree_size: 7},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: builtin_nested.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'ImplIntroducer', text: 'impl'},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'T'},
// CHECK:STDOUT:         {kind: 'TypeImplAs', text: 'as', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'IdentifierNameExpr', text: 'I'},
// CHECK:STDOUT:       {kind: 'ImplDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:           {kind: 'IdentifierNameBeforeParams', text: 'Op'},
// CHECK:STDOUT:             {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:           {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'BuiltinFunctionDefinitionStart', text: '=', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'BuiltinName', text: '"builtin"'},
// CHECK:STDOUT:       {kind: 'BuiltinFunctionDefinition', text: ';', subtree_size: 7},
// CHECK:STDOUT:     {kind: 'ImplDefinition', text: '}', subtree_size: 13},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_builtin_identifier.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'NotString'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'BuiltinFunctionDefinitionStart', text: '=', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'BuiltinFunctionDefinition', text: ';', has_error: yes, subtree_size: 6},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_builtin_two_strings.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'JunkAfterString'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'BuiltinFunctionDefinitionStart', text: '=', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'BuiltinName', text: '"hello"'},
// CHECK:STDOUT:     {kind: 'BuiltinFunctionDefinition', text: ';', has_error: yes, subtree_size: 7},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_builtin_spurious_equals.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'SpuriousEquals'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'BuiltinFunctionDefinitionStart', text: '=', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'BuiltinFunctionDefinition', text: '}', has_error: yes, subtree_size: 6},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'IdentifierNameBeforeParams', text: 'TestRecoveryFromSpuriousEquals'},
// CHECK:STDOUT:         {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:       {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
