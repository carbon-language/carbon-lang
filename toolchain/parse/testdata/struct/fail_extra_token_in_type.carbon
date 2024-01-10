// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_extra_token_in_type.carbon:[[@LINE+3]]:17: ERROR: Expected `,` or `}`.
// CHECK:STDERR: var x: {.a: i32 banana} = {.a = 0};
// CHECK:STDERR:                 ^~~~~~
var x: {.a: i32 banana} = {.a = 0};

// CHECK:STDOUT: - filename: fail_extra_token_in_type.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'x'},
// CHECK:STDOUT:           {kind: 'StructLiteralOrStructTypeLiteralStart', text: '{'},
// CHECK:STDOUT:               {kind: 'IdentifierName', text: 'a'},
// CHECK:STDOUT:             {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'StructFieldType', text: ':', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'StructTypeLiteral', text: '}', has_error: yes, subtree_size: 6},
// CHECK:STDOUT:       {kind: 'BindingPattern', text: ':', subtree_size: 8},
// CHECK:STDOUT:           {kind: 'StructLiteralOrStructTypeLiteralStart', text: '{'},
// CHECK:STDOUT:               {kind: 'IdentifierName', text: 'a'},
// CHECK:STDOUT:             {kind: 'StructFieldDesignator', text: '.', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'IntLiteral', text: '0'},
// CHECK:STDOUT:           {kind: 'StructFieldValue', text: '=', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'StructLiteral', text: '}', subtree_size: 6},
// CHECK:STDOUT:       {kind: 'VariableInitializer', text: '=', subtree_size: 7},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: ';', subtree_size: 17},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
