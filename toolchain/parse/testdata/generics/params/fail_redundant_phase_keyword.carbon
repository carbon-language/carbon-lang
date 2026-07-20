// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/generics/params/fail_redundant_phase_keyword.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/generics/params/fail_redundant_phase_keyword.carbon

// --- fail_redundant_generic_deduced.carbon

// CHECK:STDERR: fail_redundant_generic_deduced.carbon:[[@LINE+4]]:6: error: `generic` is redundant here; this binding is a checked generic by default [RedundantGenericModifier]
// CHECK:STDERR: fn F[generic T: type]();
// CHECK:STDERR:      ^~~~~~~
// CHECK:STDERR:
fn F[generic T: type]();

// --- fail_redundant_generic_class_param.carbon

// CHECK:STDERR: fail_redundant_generic_class_param.carbon:[[@LINE+4]]:9: error: `generic` is redundant here; this binding is a checked generic by default [RedundantGenericModifier]
// CHECK:STDERR: class C(generic T: type) {}
// CHECK:STDERR:         ^~~~~~~
// CHECK:STDERR:
class C(generic T: type) {}

// --- fail_redundant_runtime_explicit.carbon

// CHECK:STDERR: fail_redundant_runtime_explicit.carbon:[[@LINE+4]]:6: error: `runtime` is redundant here; this binding is runtime by default [RedundantRuntimeModifier]
// CHECK:STDERR: fn G(runtime x: i32);
// CHECK:STDERR:      ^~~~~~~
// CHECK:STDERR:
fn G(runtime x: i32);

// --- fail_redundant_generic_qualified.carbon

// CHECK:STDERR: fail_redundant_generic_qualified.carbon:[[@LINE+4]]:8: error: `generic` is redundant here; this binding is a checked generic by default [RedundantGenericModifier]
// CHECK:STDERR: fn C.F[generic T: type]();
// CHECK:STDERR:        ^~~~~~~
// CHECK:STDERR:
fn C.F[generic T: type]();

// --- fail_redundant_runtime_qualified.carbon

// CHECK:STDERR: fail_redundant_runtime_qualified.carbon:[[@LINE+4]]:8: error: `runtime` is redundant here; this binding is runtime by default [RedundantRuntimeModifier]
// CHECK:STDERR: fn C.G(runtime x: i32);
// CHECK:STDERR:        ^~~~~~~
// CHECK:STDERR:
fn C.G(runtime x: i32);

// CHECK:STDOUT: - filename: fail_redundant_generic_deduced.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'IdentifierNameMaybeBeforeSignature', text: 'F'},
// CHECK:STDOUT:         {kind: 'ImplicitParamListStart', text: '['},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeSignature', text: 'T'},
// CHECK:STDOUT:           {kind: 'CompileTimeBindingPatternStart', text: ':'},
// CHECK:STDOUT:           {kind: 'TypeTypeLiteral', text: 'type'},
// CHECK:STDOUT:         {kind: 'CompileTimeBindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'ImplicitParamList', text: ']', subtree_size: 6},
// CHECK:STDOUT:         {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:       {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 11},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_redundant_generic_class_param.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'ClassIntroducer', text: 'class'},
// CHECK:STDOUT:         {kind: 'IdentifierNameMaybeBeforeSignature', text: 'C'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'T'},
// CHECK:STDOUT:             {kind: 'CompileTimeBindingPatternStart', text: ':'},
// CHECK:STDOUT:             {kind: 'TypeTypeLiteral', text: 'type'},
// CHECK:STDOUT:           {kind: 'CompileTimeBindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 6},
// CHECK:STDOUT:       {kind: 'ClassDefinitionStart', text: '{', subtree_size: 9},
// CHECK:STDOUT:     {kind: 'ClassDefinition', text: '}', subtree_size: 10},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_redundant_runtime_explicit.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'IdentifierNameMaybeBeforeSignature', text: 'G'},
// CHECK:STDOUT:         {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'RuntimeBindingName', text: 'runtime', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'BindingPatternStart', text: ':'},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'LetBindingPattern', text: ':', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ExplicitParamList', text: ')', subtree_size: 7},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 10},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_redundant_generic_qualified.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeSignature', text: 'C'},
// CHECK:STDOUT:       {kind: 'IdentifierNameQualifierWithoutParams', text: '.', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'IdentifierNameMaybeBeforeSignature', text: 'F'},
// CHECK:STDOUT:         {kind: 'ImplicitParamListStart', text: '['},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeSignature', text: 'T'},
// CHECK:STDOUT:           {kind: 'CompileTimeBindingPatternStart', text: ':'},
// CHECK:STDOUT:           {kind: 'TypeTypeLiteral', text: 'type'},
// CHECK:STDOUT:         {kind: 'CompileTimeBindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'ImplicitParamList', text: ']', subtree_size: 6},
// CHECK:STDOUT:         {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:       {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 13},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_redundant_runtime_qualified.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeSignature', text: 'C'},
// CHECK:STDOUT:       {kind: 'IdentifierNameQualifierWithoutParams', text: '.', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'IdentifierNameMaybeBeforeSignature', text: 'G'},
// CHECK:STDOUT:         {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:           {kind: 'RuntimeBindingName', text: 'runtime', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'BindingPatternStart', text: ':'},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'LetBindingPattern', text: ':', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ExplicitParamList', text: ')', subtree_size: 7},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 12},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
