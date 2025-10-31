// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/let/let_ref.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/let/let_ref.carbon

let ref v: i32 = 0;
let ref _: i32 = 1;
fn F() {
  let ref s: str = "hello";
  let ref _: str = "goodbye";
}

// CHECK:STDOUT: - filename: let_ref.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'v'},
// CHECK:STDOUT:         {kind: 'RefBindingName', text: 'ref', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:       {kind: 'LetBindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:       {kind: 'IntLiteral', text: '0'},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 8},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:           {kind: 'UnderscoreName', text: '_'},
// CHECK:STDOUT:         {kind: 'RefBindingName', text: 'ref', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:       {kind: 'LetBindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:       {kind: 'IntLiteral', text: '1'},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 8},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'F'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeParams', text: 's'},
// CHECK:STDOUT:           {kind: 'RefBindingName', text: 'ref', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'StringTypeLiteral', text: 'str'},
// CHECK:STDOUT:         {kind: 'LetBindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'StringLiteral', text: '"hello"'},
// CHECK:STDOUT:       {kind: 'LetDecl', text: ';', subtree_size: 8},
// CHECK:STDOUT:         {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:             {kind: 'UnderscoreName', text: '_'},
// CHECK:STDOUT:           {kind: 'RefBindingName', text: 'ref', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'StringTypeLiteral', text: 'str'},
// CHECK:STDOUT:         {kind: 'LetBindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'StringLiteral', text: '"goodbye"'},
// CHECK:STDOUT:       {kind: 'LetDecl', text: ';', subtree_size: 8},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 22},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
