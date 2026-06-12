// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/generics/interface/basic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/generics/interface/basic.carbon

interface Foo {
  fn Add(self, b: Self) -> Self;
  fn Add(self: foo, b: Self) -> Self;
}

// CHECK:STDOUT: - filename: basic.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeSignature', text: 'Foo'},
// CHECK:STDOUT:       {kind: 'InterfaceDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameMaybeBeforeSignature', text: 'Add'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:             {kind: 'SelfValueName', text: 'self'},
// CHECK:STDOUT:           {kind: 'SelfBindingPattern', text: 'self', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'b'},
// CHECK:STDOUT:             {kind: 'SelfTypeNameExpr', text: 'Self'},
// CHECK:STDOUT:           {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 8},
// CHECK:STDOUT:           {kind: 'SelfTypeNameExpr', text: 'Self'},
// CHECK:STDOUT:         {kind: 'ReturnType', text: '->', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDecl', text: ';', subtree_size: 13},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameMaybeBeforeSignature', text: 'Add'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:             {kind: 'SelfValueName', text: 'self'},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'foo'},
// CHECK:STDOUT:           {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'b'},
// CHECK:STDOUT:             {kind: 'SelfTypeNameExpr', text: 'Self'},
// CHECK:STDOUT:           {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 9},
// CHECK:STDOUT:           {kind: 'SelfTypeNameExpr', text: 'Self'},
// CHECK:STDOUT:         {kind: 'ReturnType', text: '->', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDecl', text: ';', subtree_size: 14},
// CHECK:STDOUT:     {kind: 'InterfaceDefinition', text: '}', subtree_size: 31},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
