// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var n: i8 = (n)*3;

// CHECK:STDOUT: - filename: infix_with_paren_before.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:         {kind: 'Name', text: 'n'},
// CHECK:STDOUT:         {kind: 'Literal', text: 'i8'},
// CHECK:STDOUT:       {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:           {kind: 'ParenExpressionOrTupleLiteralStart', text: '('},
// CHECK:STDOUT:           {kind: 'NameExpression', text: 'n'},
// CHECK:STDOUT:         {kind: 'ParenExpression', text: ')', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'Literal', text: '3'},
// CHECK:STDOUT:       {kind: 'InfixOperator', text: '*', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'VariableDeclaration', text: ';', subtree_size: 11},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
