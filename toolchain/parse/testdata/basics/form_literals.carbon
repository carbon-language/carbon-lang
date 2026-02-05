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
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'F'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'FormLiteralKeyword', text: 'form'},
// CHECK:STDOUT:           {kind: 'FormLiteralOpenParen', text: '('},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'RefPrimitiveForm', text: 'ref', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'FormLiteral', text: ')', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 6},
// CHECK:STDOUT:           {kind: 'FormLiteralKeyword', text: 'form'},
// CHECK:STDOUT:           {kind: 'FormLiteralOpenParen', text: '('},
// CHECK:STDOUT:                 {kind: 'IdentifierNameExpr', text: 'T'},
// CHECK:STDOUT:               {kind: 'CallExprStart', text: '(', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'IdentifierNameExpr', text: 'X'},
// CHECK:STDOUT:               {kind: 'TupleLiteralComma', text: ','},
// CHECK:STDOUT:               {kind: 'IdentifierNameExpr', text: 'Y'},
// CHECK:STDOUT:             {kind: 'CallExpr', text: ')', subtree_size: 6},
// CHECK:STDOUT:           {kind: 'VarPrimitiveForm', text: 'var', subtree_size: 7},
// CHECK:STDOUT:         {kind: 'FormLiteral', text: ')', subtree_size: 10},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 11},
// CHECK:STDOUT:           {kind: 'FormLiteralKeyword', text: 'form'},
// CHECK:STDOUT:           {kind: 'FormLiteralOpenParen', text: '('},
// CHECK:STDOUT:               {kind: 'IdentifierNameExpr', text: 'X'},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeParams', text: 'Y'},
// CHECK:STDOUT:             {kind: 'MemberAccessExpr', text: '.', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'ValPrimitiveForm', text: 'val', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'FormLiteral', text: ')', subtree_size: 7},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 8},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 31},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'IdentifierNameBeforeParams', text: 'VarReturn'},
// CHECK:STDOUT:         {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:       {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'FormLiteralKeyword', text: 'form'},
// CHECK:STDOUT:           {kind: 'FormLiteralOpenParen', text: '('},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'RefPrimitiveForm', text: 'ref', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'FormLiteral', text: ')', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ReturnForm', text: '->?', subtree_size: 6},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 11},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_invalid_form_literal.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'F'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'FormLiteralKeyword', text: 'form'},
// CHECK:STDOUT:           {kind: 'FormLiteralOpenParen', text: 'form', has_error: yes},
// CHECK:STDOUT:           {kind: 'InvalidParse', text: 'form', has_error: yes},
// CHECK:STDOUT:         {kind: 'FormLiteral', text: 'form', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'FormLiteralKeyword', text: 'form'},
// CHECK:STDOUT:           {kind: 'FormLiteralOpenParen', text: '('},
// CHECK:STDOUT:             {kind: 'InvalidParse', text: ')', has_error: yes},
// CHECK:STDOUT:           {kind: 'ValPrimitiveForm', text: 'form', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:         {kind: 'FormLiteral', text: ')', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 6},
// CHECK:STDOUT:           {kind: 'FormLiteralKeyword', text: 'form'},
// CHECK:STDOUT:           {kind: 'FormLiteralOpenParen', text: '('},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'ValPrimitiveForm', text: 'form', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:         {kind: 'FormLiteral', text: ')', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 6},
// CHECK:STDOUT:           {kind: 'FormLiteralKeyword', text: 'form'},
// CHECK:STDOUT:           {kind: 'FormLiteralOpenParen', text: '('},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'whole'},
// CHECK:STDOUT:           {kind: 'ValPrimitiveForm', text: 'form', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:         {kind: 'FormLiteral', text: ')', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 6},
// CHECK:STDOUT:           {kind: 'FormLiteralKeyword', text: 'form'},
// CHECK:STDOUT:           {kind: 'FormLiteralOpenParen', text: '('},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'whole'},
// CHECK:STDOUT:           {kind: 'VarPrimitiveForm', text: 'var', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'FormLiteral', text: ')', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 6},
// CHECK:STDOUT:           {kind: 'FormLiteralKeyword', text: 'form'},
// CHECK:STDOUT:           {kind: 'FormLiteralOpenParen', text: 'form', has_error: yes},
// CHECK:STDOUT:           {kind: 'InvalidParse', text: 'form', has_error: yes},
// CHECK:STDOUT:         {kind: 'FormLiteral', text: 'form', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:           {kind: 'FormLiteralKeyword', text: 'form'},
// CHECK:STDOUT:           {kind: 'FormLiteralOpenParen', text: 'form', has_error: yes},
// CHECK:STDOUT:           {kind: 'InvalidParse', text: 'form', has_error: yes},
// CHECK:STDOUT:         {kind: 'FormLiteral', text: 'form', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 45},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
