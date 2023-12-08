// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var n: i8 = 3*(n);

// CHECK:STDOUT: - filename: infix_with_paren_after.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'n'},
// CHECK:STDOUT:         {kind: 'IntTypeLiteral', text: 'i8'},
// CHECK:STDOUT:       {kind: 'BindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'IntLiteral', text: '3'},
// CHECK:STDOUT:           {kind: 'ExprOpenParen', text: '('},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'n'},
// CHECK:STDOUT:         {kind: 'ParenExpr', text: ')', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'InfixOperator', text: '*', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: ';', subtree_size: 11},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
