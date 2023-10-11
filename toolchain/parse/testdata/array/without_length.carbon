// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var x: [i32;];

// CHECK:STDOUT: - filename: without_length.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:         {kind: 'Name', text: 'x'},
// CHECK:STDOUT:             {kind: 'ArrayExpressionStart', text: '['},
// CHECK:STDOUT:             {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:           {kind: 'ArrayExpressionSemi', text: ';', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'ArrayExpression', text: ']', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'PatternBinding', text: ':', subtree_size: 6},
// CHECK:STDOUT:     {kind: 'VariableDeclaration', text: ';', subtree_size: 8},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
