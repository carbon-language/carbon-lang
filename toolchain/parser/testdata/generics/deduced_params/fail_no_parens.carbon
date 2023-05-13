// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: [
// CHECK:STDOUT:   {kind: 'ClassIntroducer', text: 'class'},
// CHECK:STDOUT:   {kind: 'DeclaredName', text: 'Foo'},
// CHECK:STDOUT:     {kind: 'DeducedParameterListStart', text: '['},
// CHECK:STDOUT:   {kind: 'DeducedParameterList', text: ']', subtree_size: 2},
// CHECK:STDOUT: {kind: 'ClassDeclaration', text: ';', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:   {kind: 'ClassIntroducer', text: 'class'},
// CHECK:STDOUT:   {kind: 'DeclaredName', text: 'Foo'},
// CHECK:STDOUT:     {kind: 'DeducedParameterListStart', text: '['},
// CHECK:STDOUT:       {kind: 'DeclaredName', text: 'a'},
// CHECK:STDOUT:       {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:     {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:   {kind: 'DeducedParameterList', text: ']', subtree_size: 5},
// CHECK:STDOUT: {kind: 'ClassDeclaration', text: ';', has_error: yes, subtree_size: 8},
// CHECK:STDOUT:   {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:   {kind: 'DeclaredName', text: 'Bar'},
// CHECK:STDOUT:     {kind: 'DeducedParameterListStart', text: '['},
// CHECK:STDOUT:   {kind: 'DeducedParameterList', text: ']', subtree_size: 2},
// CHECK:STDOUT: {kind: 'InterfaceDeclaration', text: 'interface', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:   {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:   {kind: 'DeclaredName', text: 'Bar'},
// CHECK:STDOUT:     {kind: 'DeducedParameterListStart', text: '['},
// CHECK:STDOUT:       {kind: 'DeclaredName', text: 'a'},
// CHECK:STDOUT:       {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:     {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:   {kind: 'DeducedParameterList', text: ']', subtree_size: 5},
// CHECK:STDOUT: {kind: 'InterfaceDeclaration', text: 'interface', has_error: yes, subtree_size: 8},
// CHECK:STDOUT: {kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

// CHECK:STDERR: toolchain/parser/testdata/generics/deduced_params/fail_no_parens.carbon:[[@LINE+1]]:12: A `(` for parameters is required after deduced parameters.
class Foo[];

// CHECK:STDERR: toolchain/parser/testdata/generics/deduced_params/fail_no_parens.carbon:[[@LINE+1]]:18: A `(` for parameters is required after deduced parameters.
class Foo[a: i32];

// CHECK:STDERR: toolchain/parser/testdata/generics/deduced_params/fail_no_parens.carbon:[[@LINE+1]]:17: A `(` for parameters is required after deduced parameters.
interface Bar[] {}

// CHECK:STDERR: toolchain/parser/testdata/generics/deduced_params/fail_no_parens.carbon:[[@LINE+1]]:23: A `(` for parameters is required after deduced parameters.
interface Bar[a: i32] {}
