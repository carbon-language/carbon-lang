// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{carbon-run-parser}
// CHECK:STDOUT: [
// CHECK:STDOUT: {node_index: 26, kind: 'VariableDeclaration', text: 'var', subtree_size: 27, children: [
// CHECK:STDOUT:   {node_index: 12, kind: 'PatternBinding', text: ':', subtree_size: 13, children: [
// CHECK:STDOUT:     {node_index: 0, kind: 'DeclaredName', text: 'x'},
// CHECK:STDOUT:     {node_index: 11, kind: 'StructTypeLiteral', text: '{', subtree_size: 11, children: [
// CHECK:STDOUT:       {node_index: 4, kind: 'StructFieldType', text: ':', subtree_size: 4, children: [
// CHECK:STDOUT:         {node_index: 2, kind: 'StructFieldDesignator', text: '.', subtree_size: 2, children: [
// CHECK:STDOUT:           {node_index: 1, kind: 'DesignatedName', text: 'a'}]},
// CHECK:STDOUT:         {node_index: 3, kind: 'Literal', text: 'i32'}]},
// CHECK:STDOUT:       {node_index: 5, kind: 'StructComma', text: ','},
// CHECK:STDOUT:       {node_index: 9, kind: 'StructFieldType', text: ':', subtree_size: 4, children: [
// CHECK:STDOUT:         {node_index: 7, kind: 'StructFieldDesignator', text: '.', subtree_size: 2, children: [
// CHECK:STDOUT:           {node_index: 6, kind: 'DesignatedName', text: 'b'}]},
// CHECK:STDOUT:         {node_index: 8, kind: 'Literal', text: 'i32'}]},
// CHECK:STDOUT:       {node_index: 10, kind: 'StructEnd', text: '}'}]}]},
// CHECK:STDOUT:   {node_index: 24, kind: 'VariableInitializer', text: '=', subtree_size: 12, children: [
// CHECK:STDOUT:     {node_index: 23, kind: 'StructLiteral', text: '{', subtree_size: 11, children: [
// CHECK:STDOUT:       {node_index: 16, kind: 'StructFieldValue', text: '=', subtree_size: 4, children: [
// CHECK:STDOUT:         {node_index: 14, kind: 'StructFieldDesignator', text: '.', subtree_size: 2, children: [
// CHECK:STDOUT:           {node_index: 13, kind: 'DesignatedName', text: 'a'}]},
// CHECK:STDOUT:         {node_index: 15, kind: 'Literal', text: '1'}]},
// CHECK:STDOUT:       {node_index: 17, kind: 'StructComma', text: ','},
// CHECK:STDOUT:       {node_index: 21, kind: 'StructFieldValue', text: '=', subtree_size: 4, children: [
// CHECK:STDOUT:         {node_index: 19, kind: 'StructFieldDesignator', text: '.', subtree_size: 2, children: [
// CHECK:STDOUT:           {node_index: 18, kind: 'DesignatedName', text: 'b'}]},
// CHECK:STDOUT:         {node_index: 20, kind: 'Literal', text: '2'}]},
// CHECK:STDOUT:       {node_index: 22, kind: 'StructEnd', text: '}'}]}]},
// CHECK:STDOUT:   {node_index: 25, kind: 'DeclarationEnd', text: ';'}]},
// CHECK:STDOUT: {node_index: 35, kind: 'VariableDeclaration', text: 'var', subtree_size: 9, children: [
// CHECK:STDOUT:   {node_index: 30, kind: 'PatternBinding', text: ':', subtree_size: 4, children: [
// CHECK:STDOUT:     {node_index: 27, kind: 'DeclaredName', text: 'y'},
// CHECK:STDOUT:     {node_index: 29, kind: 'StructLiteral', text: '{', subtree_size: 2, children: [
// CHECK:STDOUT:       {node_index: 28, kind: 'StructEnd', text: '}'}]}]},
// CHECK:STDOUT:   {node_index: 33, kind: 'VariableInitializer', text: '=', subtree_size: 3, children: [
// CHECK:STDOUT:     {node_index: 32, kind: 'StructLiteral', text: '{', subtree_size: 2, children: [
// CHECK:STDOUT:       {node_index: 31, kind: 'StructEnd', text: '}'}]}]},
// CHECK:STDOUT:   {node_index: 34, kind: 'DeclarationEnd', text: ';'}]},
// CHECK:STDOUT: {node_index: 54, kind: 'VariableDeclaration', text: 'var', subtree_size: 19, children: [
// CHECK:STDOUT:   {node_index: 44, kind: 'PatternBinding', text: ':', subtree_size: 9, children: [
// CHECK:STDOUT:     {node_index: 36, kind: 'DeclaredName', text: 'z'},
// CHECK:STDOUT:     {node_index: 43, kind: 'StructTypeLiteral', text: '{', subtree_size: 7, children: [
// CHECK:STDOUT:       {node_index: 40, kind: 'StructFieldType', text: ':', subtree_size: 4, children: [
// CHECK:STDOUT:         {node_index: 38, kind: 'StructFieldDesignator', text: '.', subtree_size: 2, children: [
// CHECK:STDOUT:           {node_index: 37, kind: 'DesignatedName', text: 'n'}]},
// CHECK:STDOUT:         {node_index: 39, kind: 'Literal', text: 'i32'}]},
// CHECK:STDOUT:       {node_index: 41, kind: 'StructComma', text: ','},
// CHECK:STDOUT:       {node_index: 42, kind: 'StructEnd', text: '}'}]}]},
// CHECK:STDOUT:   {node_index: 52, kind: 'VariableInitializer', text: '=', subtree_size: 8, children: [
// CHECK:STDOUT:     {node_index: 51, kind: 'StructLiteral', text: '{', subtree_size: 7, children: [
// CHECK:STDOUT:       {node_index: 48, kind: 'StructFieldValue', text: '=', subtree_size: 4, children: [
// CHECK:STDOUT:         {node_index: 46, kind: 'StructFieldDesignator', text: '.', subtree_size: 2, children: [
// CHECK:STDOUT:           {node_index: 45, kind: 'DesignatedName', text: 'n'}]},
// CHECK:STDOUT:         {node_index: 47, kind: 'Literal', text: '4'}]},
// CHECK:STDOUT:       {node_index: 49, kind: 'StructComma', text: ','},
// CHECK:STDOUT:       {node_index: 50, kind: 'StructEnd', text: '}'}]}]},
// CHECK:STDOUT:   {node_index: 53, kind: 'DeclarationEnd', text: ';'}]},
// CHECK:STDOUT: {node_index: 55, kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

// NOTE: Move to its own directory when more tests are added.
var x: {.a: i32, .b: i32} = {.a = 1, .b = 2};
var y: {} = {};
var z: {.n: i32,} = {.n = 4,};
