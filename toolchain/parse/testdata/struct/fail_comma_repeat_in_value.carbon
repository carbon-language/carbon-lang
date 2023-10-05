// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_comma_repeat_in_value.carbon:[[@LINE+3]]:16: ERROR: Expected `.field = value`.
// CHECK:STDERR: var x: {.a = 0,,} = {};
// CHECK:STDERR:                ^
var x: {.a = 0,,} = {};

// CHECK:STDOUT: - filename: fail_comma_repeat_in_value.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:         {kind: 'Name', text: 'x'},
// CHECK:STDOUT:           {kind: 'StructLiteralOrStructTypeLiteralStart', text: '{'},
// CHECK:STDOUT:               {kind: 'Name', text: 'a'},
// CHECK:STDOUT:             {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'Literal', text: '0'},
// CHECK:STDOUT:           {kind: 'StructFieldValue', text: '=', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'StructComma', text: ','},
// CHECK:STDOUT:           {kind: 'StructFieldUnknown', text: ',', has_error: yes},
// CHECK:STDOUT:           {kind: 'StructComma', text: ','},
// CHECK:STDOUT:         {kind: 'StructLiteral', text: '}', subtree_size: 9},
// CHECK:STDOUT:       {kind: 'PatternBinding', text: ':', subtree_size: 11},
// CHECK:STDOUT:       {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'StructLiteralOrStructTypeLiteralStart', text: '{'},
// CHECK:STDOUT:       {kind: 'StructLiteral', text: '}', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'VariableDeclaration', text: ';', subtree_size: 16},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
