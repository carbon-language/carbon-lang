// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var x: (i32, i32) = (1, 2);

// CHECK:STDOUT: - filename: two_entries.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'x'},
// CHECK:STDOUT:           {kind: 'ExprOpenParen', text: '('},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'TupleLiteralComma', text: ','},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'TupleLiteral', text: ')', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'BindingPattern', text: ':', subtree_size: 7},
// CHECK:STDOUT:       {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'ExprOpenParen', text: '('},
// CHECK:STDOUT:         {kind: 'IntLiteral', text: '1'},
// CHECK:STDOUT:         {kind: 'TupleLiteralComma', text: ','},
// CHECK:STDOUT:         {kind: 'IntLiteral', text: '2'},
// CHECK:STDOUT:       {kind: 'TupleLiteral', text: ')', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: ';', subtree_size: 15},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
