// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_mix_with_unknown.carbon:[[@LINE+6]]:25: ERROR: Expected `.field = value`.
// CHECK:STDERR: var x: i32 = {.a = 1, .b, .c: i32};
// CHECK:STDERR:                         ^
// CHECK:STDERR: fail_mix_with_unknown.carbon:[[@LINE+3]]:29: ERROR: Expected `.field = value`.
// CHECK:STDERR: var x: i32 = {.a = 1, .b, .c: i32};
// CHECK:STDERR:                             ^
var x: i32 = {.a = 1, .b, .c: i32};

// CHECK:STDERR: fail_mix_with_unknown.carbon:[[@LINE+6]]:26: ERROR: Expected `.field: field_type`.
// CHECK:STDERR: var x: i32 = {.a: i32, .b, .c = 1};
// CHECK:STDERR:                          ^
// CHECK:STDERR: fail_mix_with_unknown.carbon:[[@LINE+3]]:31: ERROR: Expected `.field: field_type`.
// CHECK:STDERR: var x: i32 = {.a: i32, .b, .c = 1};
// CHECK:STDERR:                               ^
var x: i32 = {.a: i32, .b, .c = 1};

// CHECK:STDOUT: - filename: fail_mix_with_unknown.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:         {kind: 'Name', text: 'x'},
// CHECK:STDOUT:         {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:       {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'StructLiteralOrStructTypeLiteralStart', text: '{'},
// CHECK:STDOUT:             {kind: 'Name', text: 'a'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'Literal', text: '1'},
// CHECK:STDOUT:         {kind: 'StructFieldValue', text: '=', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'StructComma', text: ','},
// CHECK:STDOUT:           {kind: 'Name', text: 'b'},
// CHECK:STDOUT:         {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'StructFieldUnknown', text: '.', has_error: yes},
// CHECK:STDOUT:         {kind: 'StructComma', text: ','},
// CHECK:STDOUT:           {kind: 'Name', text: 'c'},
// CHECK:STDOUT:         {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'StructFieldUnknown', text: '.', has_error: yes},
// CHECK:STDOUT:       {kind: 'StructLiteral', text: '}', subtree_size: 14},
// CHECK:STDOUT:     {kind: 'VariableDeclaration', text: ';', subtree_size: 20},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:         {kind: 'Name', text: 'x'},
// CHECK:STDOUT:         {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:       {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'StructLiteralOrStructTypeLiteralStart', text: '{'},
// CHECK:STDOUT:             {kind: 'Name', text: 'a'},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:         {kind: 'StructFieldType', text: ':', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'StructComma', text: ','},
// CHECK:STDOUT:           {kind: 'Name', text: 'b'},
// CHECK:STDOUT:         {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'StructFieldUnknown', text: '.', has_error: yes},
// CHECK:STDOUT:         {kind: 'StructComma', text: ','},
// CHECK:STDOUT:           {kind: 'Name', text: 'c'},
// CHECK:STDOUT:         {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'StructFieldUnknown', text: '.', has_error: yes},
// CHECK:STDOUT:       {kind: 'StructTypeLiteral', text: '}', subtree_size: 14},
// CHECK:STDOUT:     {kind: 'VariableDeclaration', text: ';', subtree_size: 20},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
