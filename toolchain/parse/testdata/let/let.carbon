// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

let v: i32 = 0;
fn F() {
  let s: String = "hello";
}

// CHECK:STDOUT: - filename: let.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'v'},
// CHECK:STDOUT:         {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:       {kind: 'BindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:       {kind: 'IntLiteral', text: '0'},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 7},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'F'},
// CHECK:STDOUT:           {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:         {kind: 'TuplePattern', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:           {kind: 'IdentifierName', text: 's'},
// CHECK:STDOUT:           {kind: 'StringTypeLiteral', text: 'String'},
// CHECK:STDOUT:         {kind: 'BindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'StringLiteral', text: '"hello"'},
// CHECK:STDOUT:       {kind: 'LetDecl', text: ';', subtree_size: 7},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 13},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
