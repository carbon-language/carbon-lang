// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{carbon-run-parser}
// CHECK:STDOUT: [
// CHECK:STDOUT:   {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:     {kind: 'DeclaredName', text: 'v'},
// CHECK:STDOUT:     {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:   {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:   {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:   {kind: 'Literal', text: '0'},
// CHECK:STDOUT: {kind: 'VariableDeclaration', text: ';', subtree_size: 7},
// CHECK:STDOUT:   {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:     {kind: 'DeclaredName', text: 'w'},
// CHECK:STDOUT:     {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:   {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT: {kind: 'VariableDeclaration', text: ';', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:     {kind: 'DeclaredName', text: 'F'},
// CHECK:STDOUT:       {kind: 'ParameterListStart', text: '('},
// CHECK:STDOUT:     {kind: 'ParameterList', text: ')', subtree_size: 2},
// CHECK:STDOUT:   {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:       {kind: 'DeclaredName', text: 's'},
// CHECK:STDOUT:       {kind: 'NameReference', text: 'String'},
// CHECK:STDOUT:     {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:     {kind: 'Literal', text: '"hello"'},
// CHECK:STDOUT:   {kind: 'VariableDeclaration', text: ';', subtree_size: 7},
// CHECK:STDOUT: {kind: 'FunctionDefinition', text: '}', subtree_size: 13},
// CHECK:STDOUT: {kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

// NOTE: Move to its own directory when more tests are added.
var v: i32 = 0;
var w: i32;
fn F() {
  var s: String = "hello";
}
