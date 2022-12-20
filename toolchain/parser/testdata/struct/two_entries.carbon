// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{carbon-run-parser}
// CHECK:STDOUT: [
// CHECK:STDOUT:   {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:     {kind: 'DeclaredName', text: 'x'},
// CHECK:STDOUT:       {kind: 'StructLiteralOrStructTypeLiteralStart', text: '{'},
// CHECK:STDOUT:           {kind: 'DesignatedName', text: 'a'},
// CHECK:STDOUT:         {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:       {kind: 'StructFieldType', text: ':', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'StructComma', text: ','},
// CHECK:STDOUT:           {kind: 'DesignatedName', text: 'b'},
// CHECK:STDOUT:         {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:       {kind: 'StructFieldType', text: ':', subtree_size: 4},
// CHECK:STDOUT:     {kind: 'StructTypeLiteral', text: '}', subtree_size: 11},
// CHECK:STDOUT:   {kind: 'PatternBinding', text: ':', subtree_size: 13},
// CHECK:STDOUT:   {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:     {kind: 'StructLiteralOrStructTypeLiteralStart', text: '{'},
// CHECK:STDOUT:         {kind: 'DesignatedName', text: 'a'},
// CHECK:STDOUT:       {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'Literal', text: '1'},
// CHECK:STDOUT:     {kind: 'StructFieldValue', text: '=', subtree_size: 4},
// CHECK:STDOUT:     {kind: 'StructComma', text: ','},
// CHECK:STDOUT:         {kind: 'DesignatedName', text: 'b'},
// CHECK:STDOUT:       {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'Literal', text: '2'},
// CHECK:STDOUT:     {kind: 'StructFieldValue', text: '=', subtree_size: 4},
// CHECK:STDOUT:   {kind: 'StructLiteral', text: '}', subtree_size: 11},
// CHECK:STDOUT: {kind: 'VariableDeclaration', text: ';', subtree_size: 27},
// CHECK:STDOUT: {kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

var x: {.a: i32, .b: i32} = {.a = 1, .b = 2};
