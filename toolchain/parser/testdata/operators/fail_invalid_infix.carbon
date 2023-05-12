// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: [
// CHECK:STDOUT:   {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:     {kind: 'DeclaredName', text: 'a'},
// CHECK:STDOUT:     {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:   {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:   {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:     {kind: 'NameReference', text: 'n'},
// CHECK:STDOUT:     {kind: 'InvalidParse', text: ';', has_error: yes},
// CHECK:STDOUT:   {kind: 'InfixOperator', text: '==', has_error: yes, subtree_size: 3},
// CHECK:STDOUT: {kind: 'VariableDeclaration', text: ';', subtree_size: 9},
// CHECK:STDOUT:   {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:     {kind: 'DeclaredName', text: 'b'},
// CHECK:STDOUT:     {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:   {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:   {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:     {kind: 'InvalidParse', text: '==', has_error: yes},
// CHECK:STDOUT:     {kind: 'NameReference', text: 'n'},
// CHECK:STDOUT:   {kind: 'InfixOperator', text: '==', has_error: yes, subtree_size: 3},
// CHECK:STDOUT: {kind: 'VariableDeclaration', text: ';', subtree_size: 9},
// CHECK:STDOUT:   {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:     {kind: 'DeclaredName', text: 'c'},
// CHECK:STDOUT:     {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:   {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:   {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:     {kind: 'InvalidParse', text: '==', has_error: yes},
// CHECK:STDOUT:     {kind: 'InvalidParse', text: ';', has_error: yes},
// CHECK:STDOUT:   {kind: 'InfixOperator', text: '==', has_error: yes, subtree_size: 3},
// CHECK:STDOUT: {kind: 'VariableDeclaration', text: ';', subtree_size: 9},
// CHECK:STDOUT: {kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

// CHECK:STDERR: toolchain/parser/testdata/operators/fail_invalid_infix.carbon:[[@LINE+1]]:19: Expected expression.
var a: i32 = n == ;
// CHECK:STDERR: toolchain/parser/testdata/operators/fail_invalid_infix.carbon:[[@LINE+1]]:14: Expected expression.
var b: i32 = == n;
// CHECK:STDERR: toolchain/parser/testdata/operators/fail_invalid_infix.carbon:[[@LINE+2]]:14: Expected expression.
// CHECK:STDERR: toolchain/parser/testdata/operators/fail_invalid_infix.carbon:[[@LINE+1]]:17: Expected expression.
var c: i32 = == ;
