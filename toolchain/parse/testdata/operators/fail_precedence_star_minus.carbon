// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_precedence_star_minus.carbon:[[@LINE+3]]:16: ERROR: Parentheses are required to disambiguate operator precedence.
// CHECK:STDERR: var n: i8 = n* -n;
// CHECK:STDERR:                ^
var n: i8 = n* -n;

// CHECK:STDOUT: - filename: fail_precedence_star_minus.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'n'},
// CHECK:STDOUT:         {kind: 'IntTypeLiteral', text: 'i8'},
// CHECK:STDOUT:       {kind: 'BindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'n'},
// CHECK:STDOUT:           {kind: 'PostfixOperatorStar', text: '*', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'n'},
// CHECK:STDOUT:         {kind: 'InfixOperatorMinus', text: '-', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:       {kind: 'VariableInitializer', text: '=', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: ';', subtree_size: 10},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
