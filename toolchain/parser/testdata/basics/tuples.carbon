// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{carbon-run-parser}
// CHECK:STDOUT: [
// CHECK:STDOUT: {node_index: 14, kind: 'VariableDeclaration', text: 'var', subtree_size: 15, children: [
// CHECK:STDOUT:   {node_index: 6, kind: 'PatternBinding', text: ':', subtree_size: 7, children: [
// CHECK:STDOUT:     {node_index: 0, kind: 'DeclaredName', text: 'x'},
// CHECK:STDOUT:     {node_index: 5, kind: 'TupleLiteral', text: '(', subtree_size: 5, children: [
// CHECK:STDOUT:       {node_index: 1, kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:       {node_index: 2, kind: 'TupleLiteralComma', text: ','},
// CHECK:STDOUT:       {node_index: 3, kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:       {node_index: 4, kind: 'TupleLiteralEnd', text: ')'}]}]},
// CHECK:STDOUT:   {node_index: 12, kind: 'VariableInitializer', text: '=', subtree_size: 6, children: [
// CHECK:STDOUT:     {node_index: 11, kind: 'TupleLiteral', text: '(', subtree_size: 5, children: [
// CHECK:STDOUT:       {node_index: 7, kind: 'Literal', text: '1'},
// CHECK:STDOUT:       {node_index: 8, kind: 'TupleLiteralComma', text: ','},
// CHECK:STDOUT:       {node_index: 9, kind: 'Literal', text: '2'},
// CHECK:STDOUT:       {node_index: 10, kind: 'TupleLiteralEnd', text: ')'}]}]},
// CHECK:STDOUT:   {node_index: 13, kind: 'DeclarationEnd', text: ';'}]},
// CHECK:STDOUT: {node_index: 28, kind: 'VariableDeclaration', text: 'var', subtree_size: 14, children: [
// CHECK:STDOUT:   {node_index: 26, kind: 'PatternBinding', text: ':', subtree_size: 12, children: [
// CHECK:STDOUT:     {node_index: 15, kind: 'DeclaredName', text: 'y'},
// CHECK:STDOUT:     {node_index: 25, kind: 'TupleLiteral', text: '(', subtree_size: 10, children: [
// CHECK:STDOUT:       {node_index: 17, kind: 'TupleLiteral', text: '(', subtree_size: 2, children: [
// CHECK:STDOUT:         {node_index: 16, kind: 'TupleLiteralEnd', text: ')'}]},
// CHECK:STDOUT:       {node_index: 18, kind: 'TupleLiteralComma', text: ','},
// CHECK:STDOUT:       {node_index: 20, kind: 'TupleLiteral', text: '(', subtree_size: 2, children: [
// CHECK:STDOUT:         {node_index: 19, kind: 'TupleLiteralEnd', text: ')'}]},
// CHECK:STDOUT:       {node_index: 21, kind: 'TupleLiteralComma', text: ','},
// CHECK:STDOUT:       {node_index: 23, kind: 'TupleLiteral', text: '(', subtree_size: 2, children: [
// CHECK:STDOUT:         {node_index: 22, kind: 'TupleLiteralEnd', text: ')'}]},
// CHECK:STDOUT:       {node_index: 24, kind: 'TupleLiteralEnd', text: ')'}]}]},
// CHECK:STDOUT:   {node_index: 27, kind: 'DeclarationEnd', text: ';'}]},
// CHECK:STDOUT: {node_index: 29, kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

// NOTE: Move to its own directory when more tests are added.
var x: (i32, i32) = (1, 2);
var y: ((), (), ());
