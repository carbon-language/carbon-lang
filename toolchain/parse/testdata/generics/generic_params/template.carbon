// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/generics/generic_params/template.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/generics/generic_params/template.carbon

// --- basic.carbon

fn foo(template a:! i32);

// --- fail_template_ref.carbon

// CHECK:STDERR: fail_template_ref.carbon:[[@LINE+4]]:17: error: expected `:` binding after `ref` [ExpectedRuntimeBindingPatternAfterRef]
// CHECK:STDERR: fn foo(template ref a:! i32);
// CHECK:STDERR:                 ^~~
// CHECK:STDERR:
fn foo(template ref a:! i32);

// CHECK:STDERR: fail_template_ref.carbon:[[@LINE+4]]:8: error: expected `:!` binding after `template` [ExpectedGenericBindingPatternAfterTemplate]
// CHECK:STDERR: fn foo(template ref a: i32);
// CHECK:STDERR:        ^~~~~~~~
// CHECK:STDERR:
fn foo(template ref a: i32);

// CHECK:STDERR: fail_template_ref.carbon:[[@LINE+4]]:12: error: expected name in binding pattern [ExpectedBindingPattern]
// CHECK:STDERR: fn foo(ref template a:! i32);
// CHECK:STDERR:            ^~~~~~~~
// CHECK:STDERR:
fn foo(ref template a:! i32);

// CHECK:STDOUT: - filename: basic.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'IdentifierNameMaybeBeforeSignature', text: 'foo'},
// CHECK:STDOUT:         {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeSignature', text: 'a'},
// CHECK:STDOUT:             {kind: 'TemplateBindingName', text: 'template', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'CompileTimeBindingPatternStart', text: ':!', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'CompileTimeBindingPattern', text: ':!', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ExplicitParamList', text: ')', subtree_size: 7},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 10},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_template_ref.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'IdentifierNameMaybeBeforeSignature', text: 'foo'},
// CHECK:STDOUT:         {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:                 {kind: 'IdentifierNameNotBeforeSignature', text: 'a'},
// CHECK:STDOUT:               {kind: 'TemplateBindingName', text: 'template', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'RefBindingName', text: 'ref', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:           {kind: 'CompileTimeBindingPatternStart', text: ':!', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'CompileTimeBindingPattern', text: ':!', has_error: yes, subtree_size: 6},
// CHECK:STDOUT:       {kind: 'ExplicitParamList', text: ')', has_error: yes, subtree_size: 8},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 11},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'IdentifierNameMaybeBeforeSignature', text: 'foo'},
// CHECK:STDOUT:         {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeSignature', text: 'a'},
// CHECK:STDOUT:             {kind: 'TemplateBindingName', text: 'template', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:           {kind: 'RefBindingName', text: 'ref', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'LetBindingPattern', text: ':', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ExplicitParamList', text: ')', has_error: yes, subtree_size: 7},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 10},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'IdentifierNameMaybeBeforeSignature', text: 'foo'},
// CHECK:STDOUT:         {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeSignature', text: 'template', has_error: yes},
// CHECK:STDOUT:           {kind: 'InvalidParse', text: 'template', has_error: yes},
// CHECK:STDOUT:         {kind: 'LetBindingPattern', text: 'ref', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ExplicitParamList', text: ')', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 8},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
