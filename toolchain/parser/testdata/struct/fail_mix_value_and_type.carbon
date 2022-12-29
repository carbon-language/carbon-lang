// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{carbon-run-parser}
// CHECK:STDOUT: [
// CHECK:STDOUT:   {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:     {kind: 'DeclaredName', text: 'x'},
// CHECK:STDOUT:       {kind: 'StructLiteralOrStructTypeLiteralStart', text: '{'},
// CHECK:STDOUT:           {kind: 'DesignatedName', text: 'a'},
// CHECK:STDOUT:         {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'Literal', text: '0'},
// CHECK:STDOUT:       {kind: 'StructFieldValue', text: '=', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'StructComma', text: ','},
// CHECK:STDOUT:       {kind: 'StructFieldUnknown', text: 'b', has_error: yes},
// CHECK:STDOUT:     {kind: 'StructLiteral', text: '}', subtree_size: 8},
// CHECK:STDOUT:   {kind: 'PatternBinding', text: ':', subtree_size: 10},
// CHECK:STDOUT:   {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:     {kind: 'StructLiteralOrStructTypeLiteralStart', text: '{'},
// CHECK:STDOUT:   {kind: 'StructLiteral', text: '}', subtree_size: 2},
// CHECK:STDOUT: {kind: 'VariableDeclaration', text: ';', subtree_size: 15},
// CHECK:STDOUT: {kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

// CHECK:STDERR: {{.*}}/toolchain/parser/testdata/struct/fail_mix_value_and_type.carbon:[[@LINE+1]]:17: Expected `.field = value`.
var x: {.a = 0, b: i32} = {};
