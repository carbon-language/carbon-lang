// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: [
// CHECK:STDOUT:   {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:     {kind: 'DeclaredName', text: 'x'},
// CHECK:STDOUT:     {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:   {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:   {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:     {kind: 'StructLiteralOrStructTypeLiteralStart', text: '{'},
// CHECK:STDOUT:         {kind: 'DesignatedName', text: 'a'},
// CHECK:STDOUT:       {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'Literal', text: '1'},
// CHECK:STDOUT:     {kind: 'StructFieldValue', text: '=', subtree_size: 4},
// CHECK:STDOUT:     {kind: 'StructComma', text: ','},
// CHECK:STDOUT:       {kind: 'DesignatedName', text: 'b'},
// CHECK:STDOUT:     {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'StructFieldUnknown', text: '.', has_error: yes},
// CHECK:STDOUT:     {kind: 'StructComma', text: ','},
// CHECK:STDOUT:       {kind: 'DesignatedName', text: 'c'},
// CHECK:STDOUT:     {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'StructFieldUnknown', text: '.', has_error: yes},
// CHECK:STDOUT:   {kind: 'StructLiteral', text: '}', subtree_size: 14},
// CHECK:STDOUT: {kind: 'VariableDeclaration', text: ';', subtree_size: 20},
// CHECK:STDOUT:   {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:     {kind: 'DeclaredName', text: 'x'},
// CHECK:STDOUT:     {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:   {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:   {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:     {kind: 'StructLiteralOrStructTypeLiteralStart', text: '{'},
// CHECK:STDOUT:         {kind: 'DesignatedName', text: 'a'},
// CHECK:STDOUT:       {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:     {kind: 'StructFieldType', text: ':', subtree_size: 4},
// CHECK:STDOUT:     {kind: 'StructComma', text: ','},
// CHECK:STDOUT:       {kind: 'DesignatedName', text: 'b'},
// CHECK:STDOUT:     {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'StructFieldUnknown', text: '.', has_error: yes},
// CHECK:STDOUT:     {kind: 'StructComma', text: ','},
// CHECK:STDOUT:       {kind: 'DesignatedName', text: 'c'},
// CHECK:STDOUT:     {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'StructFieldUnknown', text: '.', has_error: yes},
// CHECK:STDOUT:   {kind: 'StructTypeLiteral', text: '}', subtree_size: 14},
// CHECK:STDOUT: {kind: 'VariableDeclaration', text: ';', subtree_size: 20},
// CHECK:STDOUT: {kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

// CHECK:STDERR: toolchain/parser/testdata/struct/fail_mix_with_unknown.carbon:[[@LINE+2]]:25: Expected `.field = value`.
// CHECK:STDERR: toolchain/parser/testdata/struct/fail_mix_with_unknown.carbon:[[@LINE+1]]:29: Expected `.field = value`.
var x: i32 = {.a = 1, .b, .c: i32};

// CHECK:STDERR: toolchain/parser/testdata/struct/fail_mix_with_unknown.carbon:[[@LINE+2]]:26: Expected `.field: field_type`.
// CHECK:STDERR: toolchain/parser/testdata/struct/fail_mix_with_unknown.carbon:[[@LINE+1]]:31: Expected `.field: field_type`.
var x: i32 = {.a: i32, .b, .c = 1};
