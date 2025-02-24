// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/function/definition/fail_builtin.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/function/definition/fail_builtin.carbon

// CHECK:STDERR: fail_builtin.carbon:[[@LINE+4]]:18: error: expected builtin function name after `=` [ExpectedBuiltinName]
// CHECK:STDERR: fn NotString() = banana;
// CHECK:STDERR:                  ^~~~~~
// CHECK:STDERR:
fn NotString() = banana;
// CHECK:STDERR: fail_builtin.carbon:[[@LINE+4]]:32: error: `fn` declarations must end with a `;` [ExpectedDeclSemi]
// CHECK:STDERR: fn JunkAfterString() = "hello" "world";
// CHECK:STDERR:                                ^~~~~~~
// CHECK:STDERR:
fn JunkAfterString() = "hello" "world";

// CHECK:STDERR: fail_builtin.carbon:[[@LINE+4]]:23: error: expected builtin function name after `=` [ExpectedBuiltinName]
// CHECK:STDERR: fn SpuriousEquals() = { }
// CHECK:STDERR:                       ^
// CHECK:STDERR:
fn SpuriousEquals() = { }
fn TestRecoveryFromSpuriousEquals();

// CHECK:STDOUT: - filename: fail_builtin.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'NotString'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'BuiltinFunctionDefinitionStart', text: '=', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'BuiltinFunctionDefinition', text: ';', has_error: yes, subtree_size: 6},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'JunkAfterString'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'BuiltinFunctionDefinitionStart', text: '=', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'BuiltinName', text: '"hello"'},
// CHECK:STDOUT:     {kind: 'BuiltinFunctionDefinition', text: ';', has_error: yes, subtree_size: 7},
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
