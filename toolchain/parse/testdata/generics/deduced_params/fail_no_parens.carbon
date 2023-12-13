// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_no_parens.carbon:[[@LINE+3]]:12: ERROR: A `(` for parameters is required after implicit parameters.
// CHECK:STDERR: class Foo[];
// CHECK:STDERR:            ^
class Foo[];

// CHECK:STDERR: fail_no_parens.carbon:[[@LINE+3]]:18: ERROR: A `(` for parameters is required after implicit parameters.
// CHECK:STDERR: class Foo[a: i32];
// CHECK:STDERR:                  ^
class Foo[a: i32];

// CHECK:STDERR: fail_no_parens.carbon:[[@LINE+3]]:17: ERROR: A `(` for parameters is required after implicit parameters.
// CHECK:STDERR: interface Bar[] {}
// CHECK:STDERR:                 ^
interface Bar[] {}

// CHECK:STDERR: fail_no_parens.carbon:[[@LINE+3]]:23: ERROR: A `(` for parameters is required after implicit parameters.
// CHECK:STDERR: interface Bar[a: i32] {}
// CHECK:STDERR:                       ^
interface Bar[a: i32] {}

// CHECK:STDOUT: - filename: fail_no_parens.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'ClassIntroducer', text: 'class'},
// CHECK:STDOUT:       {kind: 'IdentifierName', text: 'Foo'},
// CHECK:STDOUT:         {kind: 'ImplicitParamListStart', text: '['},
// CHECK:STDOUT:       {kind: 'ImplicitParamList', text: ']', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'ClassDecl', text: ';', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ClassIntroducer', text: 'class'},
// CHECK:STDOUT:       {kind: 'IdentifierName', text: 'Foo'},
// CHECK:STDOUT:         {kind: 'ImplicitParamListStart', text: '['},
// CHECK:STDOUT:           {kind: 'IdentifierName', text: 'a'},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'BindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ImplicitParamList', text: ']', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'ClassDecl', text: ';', has_error: yes, subtree_size: 8},
// CHECK:STDOUT:       {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:       {kind: 'IdentifierName', text: 'Bar'},
// CHECK:STDOUT:         {kind: 'ImplicitParamListStart', text: '['},
// CHECK:STDOUT:       {kind: 'ImplicitParamList', text: ']', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'InterfaceDecl', text: '}', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:       {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:       {kind: 'IdentifierName', text: 'Bar'},
// CHECK:STDOUT:         {kind: 'ImplicitParamListStart', text: '['},
// CHECK:STDOUT:           {kind: 'IdentifierName', text: 'a'},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'BindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ImplicitParamList', text: ']', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'InterfaceDecl', text: '}', has_error: yes, subtree_size: 8},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
