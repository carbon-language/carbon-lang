// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/where_expr/fail_rewrite.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/where_expr/fail_rewrite.carbon

// Tests for reporting errors when a rewrite requirement doesn't follow
// `where .designator = rewrite`.

// --- fail_not_designator.carbon

// CHECK:STDERR: fail_not_designator.carbon:[[@LINE+4]]:37: error: requirement can only use `=` after `.member` designator [RequirementEqualAfterNonDesignator]
// CHECK:STDERR: fn NotDesignator(T:! type where i32 = .U);
// CHECK:STDERR:                                     ^
// CHECK:STDERR:
fn NotDesignator(T:! type where i32 = .U);

// --- fail_designator_in_parens.carbon

// CHECK:STDERR: fail_designator_in_parens.carbon:[[@LINE+4]]:40: error: requirement can only use `=` after `.member` designator [RequirementEqualAfterNonDesignator]
// CHECK:STDERR: fn DesignatorInParens(V:! I where (.J) = bool);
// CHECK:STDERR:                                        ^
// CHECK:STDERR:
fn DesignatorInParens(V:! I where (.J) = bool);

// --- fail_dot_self.carbon

// CHECK:STDERR: fail_dot_self.carbon:[[@LINE+4]]:30: error: requirement can only use `=` after `.member` designator [RequirementEqualAfterNonDesignator]
// CHECK:STDERR: fn DotSelf(W:! K where .Self = f32);
// CHECK:STDERR:                              ^
// CHECK:STDERR:
fn DotSelf(W:! K where .Self = f32);

// --- fail_dot_keyword.carbon

// CHECK:STDERR: fail_dot_keyword.carbon:[[@LINE+4]]:28: error: expected identifier or `Self` after `.` [ExpectedIdentifierOrSelfAfterPeriod]
// CHECK:STDERR: fn DotKeyword(W:! K where .and = u8);
// CHECK:STDERR:                            ^~~
// CHECK:STDERR:
fn DotKeyword(W:! K where .and = u8);

// --- fail_postfix_after_designator.carbon

// CHECK:STDERR: fail_postfix_after_designator.carbon:[[@LINE+4]]:43: error: requirement can only use `=` after `.member` designator [RequirementEqualAfterNonDesignator]
// CHECK:STDERR: fn PostfixAfterDesignator(X:! L where .M* = u64*);
// CHECK:STDERR:                                           ^
// CHECK:STDERR:
fn PostfixAfterDesignator(X:! L where .M* = u64*);

// --- fail_binary_op_after_designator.carbon

// CHECK:STDERR: fail_binary_op_after_designator.carbon:[[@LINE+4]]:48: error: requirement can only use `=` after `.member` designator [RequirementEqualAfterNonDesignator]
// CHECK:STDERR: fn BinaryOpAfterDesignator(Y:! N where .O + .P = {});
// CHECK:STDERR:                                                ^
// CHECK:STDERR:
fn BinaryOpAfterDesignator(Y:! N where .O + .P = {});

// --- fail_after_and.carbon

// CHECK:STDERR: fail_after_and.carbon:[[@LINE+4]]:43: error: requirement can only use `=` after `.member` designator [RequirementEqualAfterNonDesignator]
// CHECK:STDERR: fn AfterAnd(Z:! Q where .R impls S and () = .A);
// CHECK:STDERR:                                           ^
// CHECK:STDERR:
fn AfterAnd(Z:! Q where .R impls S and () = .A);

// CHECK:STDOUT: - filename: fail_not_designator.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'IdentifierNameBeforeParams', text: 'NotDesignator'},
// CHECK:STDOUT:         {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'T'},
// CHECK:STDOUT:               {kind: 'TypeTypeLiteral', text: 'type'},
// CHECK:STDOUT:             {kind: 'WhereOperand', text: 'where', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'WhereExpr', text: 'where', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:         {kind: 'CompileTimeBindingPattern', text: ':!', has_error: yes, subtree_size: 6},
// CHECK:STDOUT:       {kind: 'ExplicitParamList', text: ')', has_error: yes, subtree_size: 8},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 11},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_designator_in_parens.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'IdentifierNameBeforeParams', text: 'DesignatorInParens'},
// CHECK:STDOUT:         {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'V'},
// CHECK:STDOUT:               {kind: 'IdentifierNameExpr', text: 'I'},
// CHECK:STDOUT:             {kind: 'WhereOperand', text: 'where', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'ParenExprStart', text: '('},
// CHECK:STDOUT:                 {kind: 'IdentifierNameNotBeforeParams', text: 'J'},
// CHECK:STDOUT:               {kind: 'DesignatorExpr', text: '.', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'ParenExpr', text: ')', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'WhereExpr', text: 'where', has_error: yes, subtree_size: 7},
// CHECK:STDOUT:         {kind: 'CompileTimeBindingPattern', text: ':!', has_error: yes, subtree_size: 9},
// CHECK:STDOUT:       {kind: 'ExplicitParamList', text: ')', has_error: yes, subtree_size: 11},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 14},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_dot_self.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'IdentifierNameBeforeParams', text: 'DotSelf'},
// CHECK:STDOUT:         {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'W'},
// CHECK:STDOUT:               {kind: 'IdentifierNameExpr', text: 'K'},
// CHECK:STDOUT:             {kind: 'WhereOperand', text: 'where', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'SelfTypeName', text: 'Self'},
// CHECK:STDOUT:             {kind: 'DesignatorExpr', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'WhereExpr', text: 'where', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:         {kind: 'CompileTimeBindingPattern', text: ':!', has_error: yes, subtree_size: 7},
// CHECK:STDOUT:       {kind: 'ExplicitParamList', text: ')', has_error: yes, subtree_size: 9},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 12},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_dot_keyword.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'IdentifierNameBeforeParams', text: 'DotKeyword'},
// CHECK:STDOUT:         {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'W'},
// CHECK:STDOUT:               {kind: 'IdentifierNameExpr', text: 'K'},
// CHECK:STDOUT:             {kind: 'WhereOperand', text: 'where', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeParams', text: 'and', has_error: yes},
// CHECK:STDOUT:             {kind: 'DesignatorExpr', text: '.', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:           {kind: 'WhereExpr', text: 'where', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:         {kind: 'CompileTimeBindingPattern', text: ':!', has_error: yes, subtree_size: 7},
// CHECK:STDOUT:       {kind: 'ExplicitParamList', text: ')', has_error: yes, subtree_size: 9},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 12},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_postfix_after_designator.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'IdentifierNameBeforeParams', text: 'PostfixAfterDesignator'},
// CHECK:STDOUT:         {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'X'},
// CHECK:STDOUT:               {kind: 'IdentifierNameExpr', text: 'L'},
// CHECK:STDOUT:             {kind: 'WhereOperand', text: 'where', subtree_size: 2},
// CHECK:STDOUT:                 {kind: 'IdentifierNameNotBeforeParams', text: 'M'},
// CHECK:STDOUT:               {kind: 'DesignatorExpr', text: '.', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'PostfixOperatorStar', text: '*', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'WhereExpr', text: 'where', has_error: yes, subtree_size: 6},
// CHECK:STDOUT:         {kind: 'CompileTimeBindingPattern', text: ':!', has_error: yes, subtree_size: 8},
// CHECK:STDOUT:       {kind: 'ExplicitParamList', text: ')', has_error: yes, subtree_size: 10},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 13},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_binary_op_after_designator.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'IdentifierNameBeforeParams', text: 'BinaryOpAfterDesignator'},
// CHECK:STDOUT:         {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'Y'},
// CHECK:STDOUT:               {kind: 'IdentifierNameExpr', text: 'N'},
// CHECK:STDOUT:             {kind: 'WhereOperand', text: 'where', subtree_size: 2},
// CHECK:STDOUT:                 {kind: 'IdentifierNameNotBeforeParams', text: 'O'},
// CHECK:STDOUT:               {kind: 'DesignatorExpr', text: '.', subtree_size: 2},
// CHECK:STDOUT:                 {kind: 'IdentifierNameNotBeforeParams', text: 'P'},
// CHECK:STDOUT:               {kind: 'DesignatorExpr', text: '.', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'InfixOperatorPlus', text: '+', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'WhereExpr', text: 'where', has_error: yes, subtree_size: 8},
// CHECK:STDOUT:         {kind: 'CompileTimeBindingPattern', text: ':!', has_error: yes, subtree_size: 10},
// CHECK:STDOUT:       {kind: 'ExplicitParamList', text: ')', has_error: yes, subtree_size: 12},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 15},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_after_and.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'IdentifierNameBeforeParams', text: 'AfterAnd'},
// CHECK:STDOUT:         {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'Z'},
// CHECK:STDOUT:               {kind: 'IdentifierNameExpr', text: 'Q'},
// CHECK:STDOUT:             {kind: 'WhereOperand', text: 'where', subtree_size: 2},
// CHECK:STDOUT:                 {kind: 'IdentifierNameNotBeforeParams', text: 'R'},
// CHECK:STDOUT:               {kind: 'DesignatorExpr', text: '.', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'IdentifierNameExpr', text: 'S'},
// CHECK:STDOUT:             {kind: 'RequirementImpls', text: 'impls', subtree_size: 4},
// CHECK:STDOUT:             {kind: 'RequirementAnd', text: 'and'},
// CHECK:STDOUT:               {kind: 'TupleLiteralStart', text: '('},
// CHECK:STDOUT:             {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'WhereExpr', text: 'where', has_error: yes, subtree_size: 10},
// CHECK:STDOUT:         {kind: 'CompileTimeBindingPattern', text: ':!', has_error: yes, subtree_size: 12},
// CHECK:STDOUT:       {kind: 'ExplicitParamList', text: ')', has_error: yes, subtree_size: 14},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 17},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
