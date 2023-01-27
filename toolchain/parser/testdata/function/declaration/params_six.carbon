// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{carbon-run-parser}
// CHECK:STDOUT: [
// CHECK:STDOUT:   {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:   {kind: 'DeclaredName', text: 'foo'},
// CHECK:STDOUT:     {kind: 'ParameterListStart', text: '('},
// CHECK:STDOUT:       {kind: 'DeclaredName', text: 'a'},
// CHECK:STDOUT:       {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:     {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'ParameterListComma', text: ','},
// CHECK:STDOUT:       {kind: 'DeclaredName', text: 'b'},
// CHECK:STDOUT:       {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:     {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'ParameterListComma', text: ','},
// CHECK:STDOUT:       {kind: 'DeclaredName', text: 'c'},
// CHECK:STDOUT:       {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:     {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'ParameterListComma', text: ','},
// CHECK:STDOUT:       {kind: 'DeclaredName', text: 'd'},
// CHECK:STDOUT:       {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:     {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'ParameterListComma', text: ','},
// CHECK:STDOUT:       {kind: 'DeclaredName', text: 'e'},
// CHECK:STDOUT:       {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:     {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'ParameterListComma', text: ','},
// CHECK:STDOUT:       {kind: 'DeclaredName', text: 'f'},
// CHECK:STDOUT:       {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:     {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:   {kind: 'ParameterList', text: ')', subtree_size: 25},
// CHECK:STDOUT: {kind: 'FunctionDeclaration', text: ';', subtree_size: 28},
// CHECK:STDOUT: {kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

fn foo(a: i32, b: i32, c: i32, d: i32, e: i32, f: i32);
