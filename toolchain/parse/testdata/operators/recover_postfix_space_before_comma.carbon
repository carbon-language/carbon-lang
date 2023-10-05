// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: recover_postfix_space_before_comma.carbon:[[@LINE+3]]:18: ERROR: Whitespace is not allowed before this unary operator.
// CHECK:STDERR: var n: i8 = F(i8 *, 0);
// CHECK:STDERR:                  ^
var n: i8 = F(i8 *, 0);

// CHECK:STDOUT: - filename: recover_postfix_space_before_comma.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:         {kind: 'Name', text: 'n'},
// CHECK:STDOUT:         {kind: 'Literal', text: 'i8'},
// CHECK:STDOUT:       {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:           {kind: 'NameExpression', text: 'F'},
// CHECK:STDOUT:         {kind: 'CallExpressionStart', text: '(', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'Literal', text: 'i8'},
// CHECK:STDOUT:         {kind: 'PostfixOperator', text: '*', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'CallExpressionComma', text: ','},
// CHECK:STDOUT:         {kind: 'Literal', text: '0'},
// CHECK:STDOUT:       {kind: 'CallExpression', text: ')', subtree_size: 7},
// CHECK:STDOUT:     {kind: 'VariableDeclaration', text: ';', subtree_size: 13},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
