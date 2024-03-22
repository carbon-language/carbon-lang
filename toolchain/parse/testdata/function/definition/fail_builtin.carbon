// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_builtin.carbon:[[@LINE+4]]:18: ERROR: Expected builtin function name after `=`.
// CHECK:STDERR: fn NotString() = banana;
// CHECK:STDERR:                  ^~~~~~
// CHECK:STDERR:
fn NotString() = banana;
// CHECK:STDERR: fail_builtin.carbon:[[@LINE+4]]:32: ERROR: `fn` declarations must end with a `;`.
// CHECK:STDERR: fn JunkAfterString() = "hello" "world";
// CHECK:STDERR:                                ^~~~~~~
// CHECK:STDERR:
fn JunkAfterString() = "hello" "world";

// CHECK:STDERR: fail_builtin.carbon:[[@LINE+3]]:23: ERROR: Expected builtin function name after `=`.
// CHECK:STDERR: fn SpuriousEquals() = { }
// CHECK:STDERR:                       ^
fn SpuriousEquals() = { }
fn TestRecoveryFromSpuriousEquals();

// CHECK:STDOUT: - filename: fail_builtin.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'NotString'},
// CHECK:STDOUT:           {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:         {kind: 'TuplePattern', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'BuiltinFunctionDefinitionStart', text: '=', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'BuiltinFunctionDefinition', text: ';', has_error: yes, subtree_size: 6},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'JunkAfterString'},
// CHECK:STDOUT:           {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:         {kind: 'TuplePattern', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'BuiltinFunctionDefinitionStart', text: '=', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'BuiltinName', text: '"hello"'},
// CHECK:STDOUT:     {kind: 'BuiltinFunctionDefinition', text: ';', has_error: yes, subtree_size: 7},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'SpuriousEquals'},
// CHECK:STDOUT:           {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:         {kind: 'TuplePattern', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'BuiltinFunctionDefinitionStart', text: '=', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'BuiltinFunctionDefinition', text: '}', has_error: yes, subtree_size: 6},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'IdentifierName', text: 'TestRecoveryFromSpuriousEquals'},
// CHECK:STDOUT:         {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:       {kind: 'TuplePattern', text: ')', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'FunctionDecl', text: ';', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
