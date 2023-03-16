// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{carbon-run-parser}
// CHECK:STDOUT: [
// CHECK:STDOUT:   {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:     {kind: 'DeclaredName', text: 'x'},
// CHECK:STDOUT:       {kind: 'ParenExpressionOrTupleLiteralStart', text: '('},
// CHECK:STDOUT:       {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:       {kind: 'TupleLiteralComma', text: ','},
// CHECK:STDOUT:       {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:     {kind: 'TupleLiteral', text: ')', subtree_size: 5},
// CHECK:STDOUT:   {kind: 'PatternBinding', text: ':', subtree_size: 7},
// CHECK:STDOUT:   {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:     {kind: 'ParenExpressionOrTupleLiteralStart', text: '('},
// CHECK:STDOUT:     {kind: 'Literal', text: '1'},
// CHECK:STDOUT:     {kind: 'TupleLiteralComma', text: ','},
// CHECK:STDOUT:     {kind: 'Literal', text: '2'},
// CHECK:STDOUT:   {kind: 'TupleLiteral', text: ')', subtree_size: 5},
// CHECK:STDOUT: {kind: 'VariableDeclaration', text: ';', subtree_size: 15},
// CHECK:STDOUT: {kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

var x: (i32, i32) = (1, 2);
