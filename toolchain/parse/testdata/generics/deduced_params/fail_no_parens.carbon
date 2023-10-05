// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_no_parens.carbon:[[@LINE+3]]:12: ERROR: A `(` for parameters is required after deduced parameters.
// CHECK:STDERR: class Foo[];
// CHECK:STDERR:            ^
class Foo[];

// CHECK:STDERR: fail_no_parens.carbon:[[@LINE+3]]:18: ERROR: A `(` for parameters is required after deduced parameters.
// CHECK:STDERR: class Foo[a: i32];
// CHECK:STDERR:                  ^
class Foo[a: i32];

// CHECK:STDERR: fail_no_parens.carbon:[[@LINE+3]]:17: ERROR: A `(` for parameters is required after deduced parameters.
// CHECK:STDERR: interface Bar[] {}
// CHECK:STDERR:                 ^
interface Bar[] {}

// CHECK:STDERR: fail_no_parens.carbon:[[@LINE+3]]:23: ERROR: A `(` for parameters is required after deduced parameters.
// CHECK:STDERR: interface Bar[a: i32] {}
// CHECK:STDERR:                       ^
interface Bar[a: i32] {}

// CHECK:STDOUT: - filename: fail_no_parens.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'ClassIntroducer', text: 'class'},
// CHECK:STDOUT:       {kind: 'Name', text: 'Foo'},
// CHECK:STDOUT:         {kind: 'DeducedParameterListStart', text: '['},
// CHECK:STDOUT:       {kind: 'DeducedParameterList', text: ']', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'ClassDeclaration', text: ';', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ClassIntroducer', text: 'class'},
// CHECK:STDOUT:       {kind: 'Name', text: 'Foo'},
// CHECK:STDOUT:         {kind: 'DeducedParameterListStart', text: '['},
// CHECK:STDOUT:           {kind: 'Name', text: 'a'},
// CHECK:STDOUT:           {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:         {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'DeducedParameterList', text: ']', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'ClassDeclaration', text: ';', has_error: yes, subtree_size: 8},
// CHECK:STDOUT:       {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:       {kind: 'Name', text: 'Bar'},
// CHECK:STDOUT:         {kind: 'DeducedParameterListStart', text: '['},
// CHECK:STDOUT:       {kind: 'DeducedParameterList', text: ']', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'InterfaceDeclaration', text: 'interface', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:       {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:       {kind: 'Name', text: 'Bar'},
// CHECK:STDOUT:         {kind: 'DeducedParameterListStart', text: '['},
// CHECK:STDOUT:           {kind: 'Name', text: 'a'},
// CHECK:STDOUT:           {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:         {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'DeducedParameterList', text: ']', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'InterfaceDeclaration', text: 'interface', has_error: yes, subtree_size: 8},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
