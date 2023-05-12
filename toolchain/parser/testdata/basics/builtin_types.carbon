// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: [
// CHECK:STDOUT:   {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:     {kind: 'DeclaredName', text: 'test_i32'},
// CHECK:STDOUT:     {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:   {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:   {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:   {kind: 'Literal', text: '0'},
// CHECK:STDOUT: {kind: 'VariableDeclaration', text: ';', subtree_size: 7},
// CHECK:STDOUT:   {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:     {kind: 'DeclaredName', text: 'test_f64'},
// CHECK:STDOUT:     {kind: 'Literal', text: 'f64'},
// CHECK:STDOUT:   {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:   {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:   {kind: 'Literal', text: '0.1'},
// CHECK:STDOUT: {kind: 'VariableDeclaration', text: ';', subtree_size: 7},
// CHECK:STDOUT:   {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:     {kind: 'DeclaredName', text: 'test_str'},
// CHECK:STDOUT:     {kind: 'Literal', text: 'String'},
// CHECK:STDOUT:   {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:   {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:   {kind: 'Literal', text: '"Test"'},
// CHECK:STDOUT: {kind: 'VariableDeclaration', text: ';', subtree_size: 7},
// CHECK:STDOUT: {kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

var test_i32: i32 = 0;
var test_f64: f64 = 0.1;
var test_str: String = "Test";
