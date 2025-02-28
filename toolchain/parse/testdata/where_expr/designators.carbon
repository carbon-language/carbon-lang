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
  // CHECK:STDERR: fail_numeric_designator.carbon:[[@LINE+4]]:4: error: expected identifier or `Self` after `.` [ExpectedIdentifierOrSelfAfterPeriod]
  // CHECK:STDERR:   .1;
  // CHECK:STDERR:    ^
  // CHECK:STDERR:
  .1;
}

// --- fail_keyword_designator.carbon

fn FailKeyword() {
  // CHECK:STDERR: fail_keyword_designator.carbon:[[@LINE+4]]:4: error: expected identifier or `Self` after `.` [ExpectedIdentifierOrSelfAfterPeriod]
  // CHECK:STDERR:   .base;
  // CHECK:STDERR:    ^~~~
  // CHECK:STDERR:
  .base;
}

// --- fail_empty_designator.carbon

fn FailEmpty() {
  // CHECK:STDERR: fail_empty_designator.carbon:[[@LINE+4]]:4: error: expected identifier or `Self` after `.` [ExpectedIdentifierOrSelfAfterPeriod]
  // CHECK:STDERR:   .;
  // CHECK:STDERR:    ^
  // CHECK:STDERR:
  .;
}

// --- fail_star_designator.carbon

fn FailStar() {
  // CHECK:STDERR: fail_star_designator.carbon:[[@LINE+4]]:4: error: expected identifier or `Self` after `.` [ExpectedIdentifierOrSelfAfterPeriod]
  // CHECK:STDERR:   .*;
  // CHECK:STDERR:    ^
  // CHECK:STDERR:
  .*;
}

// --- fail_designator_eof.carbon

var x: i32 = .

// CHECK:STDERR: fail_designator_eof.carbon:[[@LINE+95]]:21: error: expected identifier or `Self` after `.` [ExpectedIdentifierOrSelfAfterPeriod]
// CHECK:STDERR: // CHECK:STDOUT:   ]
// CHECK:STDERR:                     ^
// CHECK:STDERR:
// CHECK:STDERR: fail_designator_eof.carbon:[[@LINE+91]]:21: error: `var` declarations must end with a `;` [ExpectedDeclSemi]
// CHECK:STDERR: // CHECK:STDOUT:   ]
// CHECK:STDERR:                     ^
// CHECK:STDERR:
// CHECK:STDOUT: - filename: designator.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'Success'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'x'},
// CHECK:STDOUT:         {kind: 'DesignatorExpr', text: '.', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'SelfTypeName', text: 'Self'},
// CHECK:STDOUT:         {kind: 'DesignatorExpr', text: '.', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 12},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_numeric_designator.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'FailNumeric'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'InvalidParse', text: '1', has_error: yes},
// CHECK:STDOUT:         {kind: 'DesignatorExpr', text: '.', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 9},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_keyword_designator.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'FailKeyword'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'base', has_error: yes},
// CHECK:STDOUT:         {kind: 'DesignatorExpr', text: '.', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 9},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_empty_designator.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'FailEmpty'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'InvalidParse', text: ';', has_error: yes},
// CHECK:STDOUT:         {kind: 'DesignatorExpr', text: '.', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 9},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_star_designator.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'FailStar'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:             {kind: 'InvalidParse', text: '*', has_error: yes},
// CHECK:STDOUT:           {kind: 'DesignatorExpr', text: '.', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:         {kind: 'PostfixOperatorStar', text: '*', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 4},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 10},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_designator_eof.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'x'},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'VarBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'VariablePattern', text: 'var', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'InvalidParse', text: '', has_error: yes},
// CHECK:STDOUT:       {kind: 'DesignatorExpr', text: '.', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: '.', has_error: yes, subtree_size: 9},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
