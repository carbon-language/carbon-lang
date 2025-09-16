// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/basics/type_literals.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/basics/type_literals.carbon

var test_i32: i32 = 0;
var test_f64: f64 = 0.1;
var test_char: char = 'c';
var test_str: str = "Test";

// CHECK:STDOUT: - filename: type_literals.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'test_i32'},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'VarBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'VariablePattern', text: 'var', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:       {kind: 'IntLiteral', text: '0'},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: ';', subtree_size: 8},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'test_f64'},
// CHECK:STDOUT:           {kind: 'FloatTypeLiteral', text: 'f64'},
// CHECK:STDOUT:         {kind: 'VarBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'VariablePattern', text: 'var', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:       {kind: 'RealLiteral', text: '0.1'},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: ';', subtree_size: 8},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'test_char'},
// CHECK:STDOUT:           {kind: 'CharTypeLiteral', text: 'char'},
// CHECK:STDOUT:         {kind: 'VarBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'VariablePattern', text: 'var', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:       {kind: 'CharLiteral', text: ''c''},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: ';', subtree_size: 8},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'test_str'},
// CHECK:STDOUT:           {kind: 'StringTypeLiteral', text: 'str'},
// CHECK:STDOUT:         {kind: 'VarBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'VariablePattern', text: 'var', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:       {kind: 'StringLiteral', text: '"Test"'},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: ';', subtree_size: 8},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
