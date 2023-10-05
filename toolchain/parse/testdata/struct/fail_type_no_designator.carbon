// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_type_no_designator.carbon:[[@LINE+3]]:9: ERROR: Expected `.field: field_type` or `.field = value`.
// CHECK:STDERR: var x: {i32} = {};
// CHECK:STDERR:         ^
var x: {i32} = {};

// CHECK:STDOUT: - filename: fail_type_no_designator.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:         {kind: 'Name', text: 'x'},
// CHECK:STDOUT:           {kind: 'StructLiteralOrStructTypeLiteralStart', text: '{'},
// CHECK:STDOUT:           {kind: 'StructFieldUnknown', text: 'i32', has_error: yes},
// CHECK:STDOUT:         {kind: 'StructLiteral', text: '}', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'PatternBinding', text: ':', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'StructLiteralOrStructTypeLiteralStart', text: '{'},
// CHECK:STDOUT:       {kind: 'StructLiteral', text: '}', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'VariableDeclaration', text: ';', subtree_size: 10},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
