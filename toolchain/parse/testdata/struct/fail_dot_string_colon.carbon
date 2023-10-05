// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_dot_string_colon.carbon:[[@LINE+3]]:10: ERROR: Expected identifier after `.`.
// CHECK:STDERR: var x: {."hello": i32, .y: i32} = {};
// CHECK:STDERR:          ^
var x: {."hello": i32, .y: i32} = {};

// CHECK:STDOUT: - filename: fail_dot_string_colon.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:         {kind: 'Name', text: 'x'},
// CHECK:STDOUT:           {kind: 'StructLiteralOrStructTypeLiteralStart', text: '{'},
// CHECK:STDOUT:             {kind: 'Name', text: '"hello"', has_error: yes},
// CHECK:STDOUT:           {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:           {kind: 'StructFieldUnknown', text: ':', has_error: yes},
// CHECK:STDOUT:           {kind: 'StructComma', text: ','},
// CHECK:STDOUT:               {kind: 'Name', text: 'y'},
// CHECK:STDOUT:             {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:           {kind: 'StructFieldType', text: ':', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'StructTypeLiteral', text: '}', subtree_size: 11},
// CHECK:STDOUT:       {kind: 'PatternBinding', text: ':', subtree_size: 13},
// CHECK:STDOUT:       {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'StructLiteralOrStructTypeLiteralStart', text: '{'},
// CHECK:STDOUT:       {kind: 'StructLiteral', text: '}', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'VariableDeclaration', text: ';', subtree_size: 18},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
