// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/generics/generic_params/fail_template_no_param.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/generics/generic_params/fail_template_no_param.carbon

// CHECK:STDERR: fail_template_no_param.carbon:[[@LINE+4]]:15: error: expected name in binding pattern [ExpectedBindingPattern]
// CHECK:STDERR: fn A(template i32);
// CHECK:STDERR:               ^~~
// CHECK:STDERR:
fn A(template i32);

// CHECK:STDERR: fail_template_no_param.carbon:[[@LINE+4]]:6: error: expected `:!` binding after `template` [ExpectedGenericBindingPatternAfterTemplate]
// CHECK:STDERR: fn B(template a: i32);
// CHECK:STDERR:      ^~~~~~~~
// CHECK:STDERR:
fn B(template a: i32);

// CHECK:STDERR: fail_template_no_param.carbon:[[@LINE+4]]:14: error: expected name in binding pattern [ExpectedBindingPattern]
// CHECK:STDERR: fn C(template);
// CHECK:STDERR:              ^
// CHECK:STDERR:
fn C(template);

// CHECK:STDERR: fail_template_no_param.carbon:[[@LINE+4]]:14: error: expected name in binding pattern [ExpectedBindingPattern]
// CHECK:STDERR: fn D(template:! i32);
// CHECK:STDERR:              ^~
// CHECK:STDERR:
fn D(template:! i32);

// CHECK:STDERR: fail_template_no_param.carbon:[[@LINE+4]]:14: error: expected name in binding pattern [ExpectedBindingPattern]
// CHECK:STDERR: fn E(template:! i32);
// CHECK:STDERR:              ^~
// CHECK:STDERR:
fn E(template:! i32);

// CHECK:STDOUT: - filename: fail_template_no_param.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'IdentifierNameBeforeParams', text: 'A'},
// CHECK:STDOUT:         {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'i32', has_error: yes},
// CHECK:STDOUT:           {kind: 'InvalidParse', text: 'i32', has_error: yes},
// CHECK:STDOUT:         {kind: 'LetBindingPattern', text: 'template', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ExplicitParamList', text: ')', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 8},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'IdentifierNameBeforeParams', text: 'B'},
// CHECK:STDOUT:         {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeParams', text: 'a'},
// CHECK:STDOUT:           {kind: 'TemplateBindingName', text: 'template', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'LetBindingPattern', text: ':', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:       {kind: 'ExplicitParamList', text: ')', has_error: yes, subtree_size: 6},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 9},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'IdentifierNameBeforeParams', text: 'C'},
// CHECK:STDOUT:         {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: ')', has_error: yes},
// CHECK:STDOUT:           {kind: 'InvalidParse', text: ')', has_error: yes},
// CHECK:STDOUT:         {kind: 'LetBindingPattern', text: 'template', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ExplicitParamList', text: ')', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 8},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'IdentifierNameBeforeParams', text: 'D'},
// CHECK:STDOUT:         {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeParams', text: ':!', has_error: yes},
// CHECK:STDOUT:           {kind: 'TemplateBindingName', text: 'template', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'CompileTimeBindingPattern', text: ':!', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:       {kind: 'ExplicitParamList', text: ')', has_error: yes, subtree_size: 6},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 9},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'IdentifierNameBeforeParams', text: 'E'},
// CHECK:STDOUT:         {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeParams', text: ':!', has_error: yes},
// CHECK:STDOUT:           {kind: 'TemplateBindingName', text: 'template', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'CompileTimeBindingPattern', text: ':!', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:       {kind: 'ExplicitParamList', text: ')', has_error: yes, subtree_size: 6},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 9},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
