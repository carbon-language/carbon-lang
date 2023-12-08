// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var y: ((), (), ());

// CHECK:STDOUT: - filename: nested.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'y'},
// CHECK:STDOUT:           {kind: 'ExprOpenParen', text: '('},
// CHECK:STDOUT:             {kind: 'ExprOpenParen', text: '('},
// CHECK:STDOUT:           {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'TupleLiteralComma', text: ','},
// CHECK:STDOUT:             {kind: 'ExprOpenParen', text: '('},
// CHECK:STDOUT:           {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'TupleLiteralComma', text: ','},
// CHECK:STDOUT:             {kind: 'ExprOpenParen', text: '('},
// CHECK:STDOUT:           {kind: 'TupleLiteral', text: ')', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'TupleLiteral', text: ')', subtree_size: 10},
// CHECK:STDOUT:       {kind: 'BindingPattern', text: ':', subtree_size: 12},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: ';', subtree_size: 14},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
