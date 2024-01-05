// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_empty_expr.carbon:[[@LINE+3]]:16: ERROR: Expected expression.
// CHECK:STDERR: var v: i32 = t[];
// CHECK:STDERR:                ^
var v: i32 = t[];

// CHECK:STDOUT: - filename: fail_empty_expr.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'v'},
// CHECK:STDOUT:         {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:       {kind: 'BindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 't'},
// CHECK:STDOUT:           {kind: 'IndexExprStart', text: '[', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'InvalidParse', text: ']', has_error: yes},
// CHECK:STDOUT:         {kind: 'IndexExpr', text: ']', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:       {kind: 'VariableInitializer', text: '=', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: ';', subtree_size: 10},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
