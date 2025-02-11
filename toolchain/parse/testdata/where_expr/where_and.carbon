// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/where_expr/where_and.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/where_expr/where_and.carbon

// --- success.carbon

fn OneAnd(T:! type where .U = i32 and .V == .W);

fn TwoAnd(Y:! I where .J impls K and .L == .M and .N = bool);

// --- fail_and_prefix.carbon

// CHECK:STDERR: fail_and_prefix.carbon:[[@LINE+4]]:29: error: expected expression [ExpectedExpr]
// CHECK:STDERR: fn AndPrefix(T:! type where and .V == .W);
// CHECK:STDERR:                             ^~~
// CHECK:STDERR:
fn AndPrefix(T:! type where and .V == .W);

// --- fail_and_suffix.carbon

// CHECK:STDERR: fail_and_suffix.carbon:[[@LINE+4]]:40: error: expected expression [ExpectedExpr]
// CHECK:STDERR: fn AndSuffix(Y:! I where .J impls K and);
// CHECK:STDERR:                                        ^
// CHECK:STDERR:
fn AndSuffix(Y:! I where .J impls K and);

// --- fail_and_early.carbon

// CHECK:STDERR: fail_and_early.carbon:[[@LINE+4]]:28: error: requirement should use `impls`, `=`, or `==` operator [ExpectedRequirementOperator]
// CHECK:STDERR: fn AndEarly(Z:! L where .M and N);
// CHECK:STDERR:                            ^~~
// CHECK:STDERR:
fn AndEarly(Z:! L where .M and N);

// CHECK:STDOUT: - filename: success.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'IdentifierNameBeforeParams', text: 'OneAnd'},
// CHECK:STDOUT:         {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'T'},
// CHECK:STDOUT:               {kind: 'TypeTypeLiteral', text: 'type'},
// CHECK:STDOUT:             {kind: 'WhereOperand', text: 'where', subtree_size: 2},
// CHECK:STDOUT:                 {kind: 'IdentifierNameNotBeforeParams', text: 'U'},
// CHECK:STDOUT:               {kind: 'DesignatorExpr', text: '.', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:             {kind: 'RequirementEqual', text: '=', subtree_size: 4},
// CHECK:STDOUT:             {kind: 'RequirementAnd', text: 'and'},
// CHECK:STDOUT:                 {kind: 'IdentifierNameNotBeforeParams', text: 'V'},
// CHECK:STDOUT:               {kind: 'DesignatorExpr', text: '.', subtree_size: 2},
// CHECK:STDOUT:                 {kind: 'IdentifierNameNotBeforeParams', text: 'W'},
// CHECK:STDOUT:               {kind: 'DesignatorExpr', text: '.', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'RequirementEqualEqual', text: '==', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'WhereExpr', text: 'where', subtree_size: 13},
// CHECK:STDOUT:         {kind: 'CompileTimeBindingPattern', text: ':!', subtree_size: 15},
// CHECK:STDOUT:       {kind: 'ExplicitParamList', text: ')', subtree_size: 17},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 20},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'IdentifierNameBeforeParams', text: 'TwoAnd'},
// CHECK:STDOUT:         {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'Y'},
// CHECK:STDOUT:               {kind: 'IdentifierNameExpr', text: 'I'},
// CHECK:STDOUT:             {kind: 'WhereOperand', text: 'where', subtree_size: 2},
// CHECK:STDOUT:                 {kind: 'IdentifierNameNotBeforeParams', text: 'J'},
// CHECK:STDOUT:               {kind: 'DesignatorExpr', text: '.', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'IdentifierNameExpr', text: 'K'},
// CHECK:STDOUT:             {kind: 'RequirementImpls', text: 'impls', subtree_size: 4},
// CHECK:STDOUT:             {kind: 'RequirementAnd', text: 'and'},
// CHECK:STDOUT:                 {kind: 'IdentifierNameNotBeforeParams', text: 'L'},
// CHECK:STDOUT:               {kind: 'DesignatorExpr', text: '.', subtree_size: 2},
// CHECK:STDOUT:                 {kind: 'IdentifierNameNotBeforeParams', text: 'M'},
// CHECK:STDOUT:               {kind: 'DesignatorExpr', text: '.', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'RequirementEqualEqual', text: '==', subtree_size: 5},
// CHECK:STDOUT:             {kind: 'RequirementAnd', text: 'and'},
// CHECK:STDOUT:                 {kind: 'IdentifierNameNotBeforeParams', text: 'N'},
// CHECK:STDOUT:               {kind: 'DesignatorExpr', text: '.', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'BoolTypeLiteral', text: 'bool'},
// CHECK:STDOUT:             {kind: 'RequirementEqual', text: '=', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'WhereExpr', text: 'where', subtree_size: 18},
// CHECK:STDOUT:         {kind: 'CompileTimeBindingPattern', text: ':!', subtree_size: 20},
// CHECK:STDOUT:       {kind: 'ExplicitParamList', text: ')', subtree_size: 22},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 25},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_and_prefix.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'IdentifierNameBeforeParams', text: 'AndPrefix'},
// CHECK:STDOUT:         {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'T'},
// CHECK:STDOUT:               {kind: 'TypeTypeLiteral', text: 'type'},
// CHECK:STDOUT:             {kind: 'WhereOperand', text: 'where', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'InvalidParse', text: 'and', has_error: yes},
// CHECK:STDOUT:           {kind: 'WhereExpr', text: 'where', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:         {kind: 'CompileTimeBindingPattern', text: ':!', has_error: yes, subtree_size: 6},
// CHECK:STDOUT:       {kind: 'ExplicitParamList', text: ')', has_error: yes, subtree_size: 8},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 11},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_and_suffix.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'IdentifierNameBeforeParams', text: 'AndSuffix'},
// CHECK:STDOUT:         {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'Y'},
// CHECK:STDOUT:               {kind: 'IdentifierNameExpr', text: 'I'},
// CHECK:STDOUT:             {kind: 'WhereOperand', text: 'where', subtree_size: 2},
// CHECK:STDOUT:                 {kind: 'IdentifierNameNotBeforeParams', text: 'J'},
// CHECK:STDOUT:               {kind: 'DesignatorExpr', text: '.', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'IdentifierNameExpr', text: 'K'},
// CHECK:STDOUT:             {kind: 'RequirementImpls', text: 'impls', subtree_size: 4},
// CHECK:STDOUT:             {kind: 'RequirementAnd', text: 'and'},
// CHECK:STDOUT:             {kind: 'InvalidParse', text: ')', has_error: yes},
// CHECK:STDOUT:           {kind: 'WhereExpr', text: 'where', has_error: yes, subtree_size: 9},
// CHECK:STDOUT:         {kind: 'CompileTimeBindingPattern', text: ':!', has_error: yes, subtree_size: 11},
// CHECK:STDOUT:       {kind: 'ExplicitParamList', text: ')', has_error: yes, subtree_size: 13},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 16},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_and_early.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'IdentifierNameBeforeParams', text: 'AndEarly'},
// CHECK:STDOUT:         {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'Z'},
// CHECK:STDOUT:               {kind: 'IdentifierNameExpr', text: 'L'},
// CHECK:STDOUT:             {kind: 'WhereOperand', text: 'where', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeParams', text: 'M'},
// CHECK:STDOUT:             {kind: 'DesignatorExpr', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'WhereExpr', text: 'where', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:         {kind: 'CompileTimeBindingPattern', text: ':!', has_error: yes, subtree_size: 7},
// CHECK:STDOUT:       {kind: 'ExplicitParamList', text: ')', has_error: yes, subtree_size: 9},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 12},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
