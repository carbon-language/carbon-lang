// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/array/fail_syntax.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/array/fail_syntax.carbon

// --- fail_no_comma.carbon

// CHECK:STDERR: fail_no_comma.carbon:[[@LINE+8]]:17: error: expected `,` in `array(Type, Count)` [ExpectedArrayComma]
// CHECK:STDERR: var x: array(i32);
// CHECK:STDERR:                 ^
// CHECK:STDERR:
// CHECK:STDERR: fail_no_comma.carbon:[[@LINE+4]]:17: error: expected expression [ExpectedExpr]
// CHECK:STDERR: var x: array(i32);
// CHECK:STDERR:                 ^
// CHECK:STDERR:
var x: array(i32);

// --- fail_invalid_char.carbon

// CHECK:STDERR: fail_invalid_char.carbon:[[@LINE+16]]:17: error: expected expression [ExpectedExpr]
// CHECK:STDERR: fn X() -> array(:);
// CHECK:STDERR:                 ^
// CHECK:STDERR:
// CHECK:STDERR: fail_invalid_char.carbon:[[@LINE+12]]:17: error: expected `,` in `array(Type, Count)` [ExpectedArrayComma]
// CHECK:STDERR: fn X() -> array(:);
// CHECK:STDERR:                 ^
// CHECK:STDERR:
// CHECK:STDERR: fail_invalid_char.carbon:[[@LINE+8]]:17: error: expected expression [ExpectedExpr]
// CHECK:STDERR: fn X() -> array(:);
// CHECK:STDERR:                 ^
// CHECK:STDERR:
// CHECK:STDERR: fail_invalid_char.carbon:[[@LINE+4]]:17: error: unexpected tokens before `)` [ExpectedCloseSymbol]
// CHECK:STDERR: fn X() -> array(:);
// CHECK:STDERR:                 ^
// CHECK:STDERR:
fn X() -> array(:);

// --- fail_unlexed_expr.carbon

// CHECK:STDERR: fail_unlexed_expr.carbon:[[@LINE+12]]:14: error: encountered unrecognized characters while parsing [UnrecognizedCharacters]
// CHECK:STDERR: var y: array(`);
// CHECK:STDERR:              ^
// CHECK:STDERR:
// CHECK:STDERR: fail_unlexed_expr.carbon:[[@LINE+8]]:14: error: expected `,` in `array(Type, Count)` [ExpectedArrayComma]
// CHECK:STDERR: var y: array(`);
// CHECK:STDERR:              ^
// CHECK:STDERR:
// CHECK:STDERR: fail_unlexed_expr.carbon:[[@LINE+4]]:14: error: unexpected tokens before `)` [ExpectedCloseSymbol]
// CHECK:STDERR: var y: array(`);
// CHECK:STDERR:              ^
// CHECK:STDERR:
var y: array(`);

// --- fail_no_open_paren.carbon

// CHECK:STDERR: fail_no_open_paren.carbon:[[@LINE+4]]:14: error: expected `(` after `array` [ExpectedParenAfter]
// CHECK:STDERR: var x: array i32, 1;
// CHECK:STDERR:              ^~~
// CHECK:STDERR:
var x: array i32, 1;

// --- fail_no_close_paren.carbon

// CHECK:STDERR: fail_no_close_paren.carbon:[[@LINE+16]]:13: error: opening symbol without a corresponding closing symbol [UnmatchedOpening]
// CHECK:STDERR: var x: array(i32,;
// CHECK:STDERR:             ^
// CHECK:STDERR:
// CHECK:STDERR: fail_no_close_paren.carbon:[[@LINE+12]]:13: error: expected `(` after `array` [ExpectedParenAfter]
// CHECK:STDERR: var x: array(i32,;
// CHECK:STDERR:             ^
// CHECK:STDERR:
// CHECK:STDERR: fail_no_close_paren.carbon:[[@LINE+8]]:13: error: expected `,` in `array(Type, Count)` [ExpectedArrayComma]
// CHECK:STDERR: var x: array(i32,;
// CHECK:STDERR:             ^
// CHECK:STDERR:
// CHECK:STDERR: fail_no_close_paren.carbon:[[@LINE+4]]:13: error: `var` declarations must end with a `;` [ExpectedDeclSemi]
// CHECK:STDERR: var x: array(i32,;
// CHECK:STDERR:             ^
// CHECK:STDERR:
var x: array(i32,;

// --- fail_no_length.carbon

// CHECK:STDERR: fail_no_length.carbon:[[@LINE+4]]:18: error: expected expression [ExpectedExpr]
// CHECK:STDERR: var x: array(i32,);
// CHECK:STDERR:                  ^
// CHECK:STDERR:
var x: array(i32,);

// --- fail_no_params.carbon

// CHECK:STDERR: fail_no_params.carbon:[[@LINE+12]]:14: error: expected expression [ExpectedExpr]
// CHECK:STDERR: var x: array();
// CHECK:STDERR:              ^
// CHECK:STDERR:
// CHECK:STDERR: fail_no_params.carbon:[[@LINE+8]]:14: error: expected `,` in `array(Type, Count)` [ExpectedArrayComma]
// CHECK:STDERR: var x: array();
// CHECK:STDERR:              ^
// CHECK:STDERR:
// CHECK:STDERR: fail_no_params.carbon:[[@LINE+4]]:14: error: expected expression [ExpectedExpr]
// CHECK:STDERR: var x: array();
// CHECK:STDERR:              ^
// CHECK:STDERR:
var x: array();

// --- fail_keyword_alone.carbon

// CHECK:STDERR: fail_keyword_alone.carbon:[[@LINE+16]]:13: error: expected `(` after `array` [ExpectedParenAfter]
// CHECK:STDERR: var x: array;
// CHECK:STDERR:             ^
// CHECK:STDERR:
// CHECK:STDERR: fail_keyword_alone.carbon:[[@LINE+12]]:13: error: expected expression [ExpectedExpr]
// CHECK:STDERR: var x: array;
// CHECK:STDERR:             ^
// CHECK:STDERR:
// CHECK:STDERR: fail_keyword_alone.carbon:[[@LINE+8]]:13: error: expected `,` in `array(Type, Count)` [ExpectedArrayComma]
// CHECK:STDERR: var x: array;
// CHECK:STDERR:             ^
// CHECK:STDERR:
// CHECK:STDERR: fail_keyword_alone.carbon:[[@LINE+4]]:13: error: expected expression [ExpectedExpr]
// CHECK:STDERR: var x: array;
// CHECK:STDERR:             ^
// CHECK:STDERR:
var x: array;

// CHECK:STDOUT: - filename: fail_no_comma.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'x'},
// CHECK:STDOUT:             {kind: 'ArrayExprKeyword', text: 'array'},
// CHECK:STDOUT:             {kind: 'ArrayExprOpenParen', text: '('},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:             {kind: 'ArrayExprComma', text: ')', has_error: yes},
// CHECK:STDOUT:             {kind: 'InvalidParse', text: ')', has_error: yes},
// CHECK:STDOUT:           {kind: 'ArrayExpr', text: ')', has_error: yes, subtree_size: 6},
// CHECK:STDOUT:         {kind: 'VarBindingPattern', text: ':', subtree_size: 8},
// CHECK:STDOUT:       {kind: 'VariablePattern', text: 'var', subtree_size: 9},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: ';', subtree_size: 11},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_invalid_char.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'IdentifierNameBeforeParams', text: 'X'},
// CHECK:STDOUT:         {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:       {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'ArrayExprKeyword', text: 'array'},
// CHECK:STDOUT:           {kind: 'ArrayExprOpenParen', text: '('},
// CHECK:STDOUT:           {kind: 'InvalidParse', text: ':', has_error: yes},
// CHECK:STDOUT:           {kind: 'ArrayExprComma', text: ':', has_error: yes},
// CHECK:STDOUT:           {kind: 'InvalidParse', text: ':', has_error: yes},
// CHECK:STDOUT:         {kind: 'ArrayExpr', text: ')', has_error: yes, subtree_size: 6},
// CHECK:STDOUT:       {kind: 'ReturnType', text: '->', subtree_size: 7},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 12},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_unlexed_expr.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'y'},
// CHECK:STDOUT:             {kind: 'ArrayExprKeyword', text: 'array'},
// CHECK:STDOUT:             {kind: 'ArrayExprOpenParen', text: '('},
// CHECK:STDOUT:             {kind: 'InvalidParse', text: '`', has_error: yes},
// CHECK:STDOUT:             {kind: 'ArrayExprComma', text: '`', has_error: yes},
// CHECK:STDOUT:             {kind: 'InvalidParse', text: '`', has_error: yes},
// CHECK:STDOUT:           {kind: 'ArrayExpr', text: ')', has_error: yes, subtree_size: 6},
// CHECK:STDOUT:         {kind: 'VarBindingPattern', text: ':', subtree_size: 8},
// CHECK:STDOUT:       {kind: 'VariablePattern', text: 'var', subtree_size: 9},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: ';', subtree_size: 11},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_no_open_paren.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'x'},
// CHECK:STDOUT:             {kind: 'ArrayExprKeyword', text: 'array'},
// CHECK:STDOUT:             {kind: 'ArrayExprOpenParen', text: 'array', has_error: yes},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:             {kind: 'ArrayExprComma', text: ','},
// CHECK:STDOUT:             {kind: 'IntLiteral', text: '1'},
// CHECK:STDOUT:           {kind: 'ArrayExpr', text: 'array', has_error: yes, subtree_size: 6},
// CHECK:STDOUT:         {kind: 'VarBindingPattern', text: ':', subtree_size: 8},
// CHECK:STDOUT:       {kind: 'VariablePattern', text: 'var', subtree_size: 9},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: ';', subtree_size: 11},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_no_close_paren.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'x'},
// CHECK:STDOUT:             {kind: 'ArrayExprKeyword', text: 'array'},
// CHECK:STDOUT:             {kind: 'ArrayExprOpenParen', text: 'array', has_error: yes},
// CHECK:STDOUT:             {kind: 'InvalidParse', text: '(', has_error: yes},
// CHECK:STDOUT:             {kind: 'ArrayExprComma', text: '(', has_error: yes},
// CHECK:STDOUT:             {kind: 'InvalidParse', text: '(', has_error: yes},
// CHECK:STDOUT:           {kind: 'ArrayExpr', text: 'array', has_error: yes, subtree_size: 6},
// CHECK:STDOUT:         {kind: 'VarBindingPattern', text: ':', subtree_size: 8},
// CHECK:STDOUT:       {kind: 'VariablePattern', text: 'var', subtree_size: 9},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: ';', has_error: yes, subtree_size: 11},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_no_length.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'x'},
// CHECK:STDOUT:             {kind: 'ArrayExprKeyword', text: 'array'},
// CHECK:STDOUT:             {kind: 'ArrayExprOpenParen', text: '('},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:             {kind: 'ArrayExprComma', text: ','},
// CHECK:STDOUT:             {kind: 'InvalidParse', text: ')', has_error: yes},
// CHECK:STDOUT:           {kind: 'ArrayExpr', text: ')', has_error: yes, subtree_size: 6},
// CHECK:STDOUT:         {kind: 'VarBindingPattern', text: ':', subtree_size: 8},
// CHECK:STDOUT:       {kind: 'VariablePattern', text: 'var', subtree_size: 9},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: ';', subtree_size: 11},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_no_params.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'x'},
// CHECK:STDOUT:             {kind: 'ArrayExprKeyword', text: 'array'},
// CHECK:STDOUT:             {kind: 'ArrayExprOpenParen', text: '('},
// CHECK:STDOUT:             {kind: 'InvalidParse', text: ')', has_error: yes},
// CHECK:STDOUT:             {kind: 'ArrayExprComma', text: ')', has_error: yes},
// CHECK:STDOUT:             {kind: 'InvalidParse', text: ')', has_error: yes},
// CHECK:STDOUT:           {kind: 'ArrayExpr', text: ')', has_error: yes, subtree_size: 6},
// CHECK:STDOUT:         {kind: 'VarBindingPattern', text: ':', subtree_size: 8},
// CHECK:STDOUT:       {kind: 'VariablePattern', text: 'var', subtree_size: 9},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: ';', subtree_size: 11},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_keyword_alone.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'x'},
// CHECK:STDOUT:             {kind: 'ArrayExprKeyword', text: 'array'},
// CHECK:STDOUT:             {kind: 'ArrayExprOpenParen', text: 'array', has_error: yes},
// CHECK:STDOUT:             {kind: 'InvalidParse', text: ';', has_error: yes},
// CHECK:STDOUT:             {kind: 'ArrayExprComma', text: ';', has_error: yes},
// CHECK:STDOUT:             {kind: 'InvalidParse', text: ';', has_error: yes},
// CHECK:STDOUT:           {kind: 'ArrayExpr', text: 'array', has_error: yes, subtree_size: 6},
// CHECK:STDOUT:         {kind: 'VarBindingPattern', text: ':', subtree_size: 8},
// CHECK:STDOUT:       {kind: 'VariablePattern', text: 'var', subtree_size: 9},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: ';', subtree_size: 11},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
