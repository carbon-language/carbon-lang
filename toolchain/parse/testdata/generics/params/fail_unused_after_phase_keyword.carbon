// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/generics/params/fail_unused_after_phase_keyword.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/generics/params/fail_unused_after_phase_keyword.carbon

// --- fail_generic_before_unused.carbon

// CHECK:STDERR: fail_generic_before_unused.carbon:[[@LINE+4]]:14: error: `unused` must be written before `generic` [UnusedAfterBindingModifier]
// CHECK:STDERR: fn F(generic unused T: type);
// CHECK:STDERR:              ^~~~~~
// CHECK:STDERR:
fn F(generic unused T: type);

// --- fail_runtime_before_unused.carbon

// CHECK:STDERR: fail_runtime_before_unused.carbon:[[@LINE+4]]:14: error: `unused` must be written before `runtime` [UnusedAfterBindingModifier]
// CHECK:STDERR: fn G[runtime unused T: type]();
// CHECK:STDERR:              ^~~~~~
// CHECK:STDERR:
fn G[runtime unused T: type]();

// --- fail_redundant_runtime_before_unused.carbon

// CHECK:STDERR: fail_redundant_runtime_before_unused.carbon:[[@LINE+4]]:6: error: `runtime` is redundant here; this binding is runtime by default [RedundantRuntimeModifier]
// CHECK:STDERR: fn H(runtime unused x: i32);
// CHECK:STDERR:      ^~~~~~~
// CHECK:STDERR:
fn H(runtime unused x: i32);

// CHECK:STDOUT: - filename: fail_generic_before_unused.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'IdentifierNameMaybeBeforeSignature', text: 'F'},
// CHECK:STDOUT:         {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'T'},
// CHECK:STDOUT:             {kind: 'CompileTimeBindingPatternTypeStart', text: ':', has_error: yes},
// CHECK:STDOUT:             {kind: 'TypeTypeLiteral', text: 'type'},
// CHECK:STDOUT:           {kind: 'CompileTimeBindingPattern', text: ':', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:         {kind: 'UnusedPattern', text: 'unused', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ExplicitParamList', text: ')', has_error: yes, subtree_size: 7},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 10},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_runtime_before_unused.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'IdentifierNameMaybeBeforeSignature', text: 'G'},
// CHECK:STDOUT:         {kind: 'ImplicitParamListStart', text: '['},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeSignature', text: 'T'},
// CHECK:STDOUT:             {kind: 'RuntimeBindingName', text: 'runtime', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'BindingPatternTypeStart', text: ':', has_error: yes},
// CHECK:STDOUT:             {kind: 'TypeTypeLiteral', text: 'type'},
// CHECK:STDOUT:           {kind: 'LetBindingPattern', text: ':', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:         {kind: 'UnusedPattern', text: 'unused', has_error: yes, subtree_size: 6},
// CHECK:STDOUT:       {kind: 'ImplicitParamList', text: ']', has_error: yes, subtree_size: 8},
// CHECK:STDOUT:         {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:       {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 13},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_redundant_runtime_before_unused.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'IdentifierNameMaybeBeforeSignature', text: 'H'},
// CHECK:STDOUT:         {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:             {kind: 'RuntimeBindingName', text: 'runtime', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'BindingPatternTypeStart', text: ':'},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'LetBindingPattern', text: ':', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'UnusedPattern', text: 'unused', subtree_size: 6},
// CHECK:STDOUT:       {kind: 'ExplicitParamList', text: ')', subtree_size: 8},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 11},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
