// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/let/missing_value.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/let/missing_value.carbon

let a: i32;

fn F() {
  let b: i32;
}

interface I {
  let T:! type;
}

// CHECK:STDOUT: - filename: missing_value.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'a'},
// CHECK:STDOUT:         {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:       {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'LetDecl', text: ';', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'F'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'b'},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'LetDecl', text: ';', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 11},
// CHECK:STDOUT:         {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'I'},
// CHECK:STDOUT:       {kind: 'InterfaceDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'T'},
// CHECK:STDOUT:           {kind: 'TypeTypeLiteral', text: 'type'},
// CHECK:STDOUT:         {kind: 'CompileTimeBindingPattern', text: ':!', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'LetDecl', text: ';', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'InterfaceDefinition', text: '}', subtree_size: 9},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
