// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/generics/named_constraint/invalid_method.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/generics/named_constraint/invalid_method.carbon

constraint E {
  // This is an error in the check phase.
  fn DeclaredMethod(self, b: Self) -> Self;
}

constraint Foo {
  // This is an error in the check phase.
  fn DefinedMethod(unused self, b: Self) -> Self {}
}

// CHECK:STDOUT: - filename: invalid_method.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'NamedConstraintIntroducer', text: 'constraint'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeSignature', text: 'E'},
// CHECK:STDOUT:       {kind: 'NamedConstraintDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameMaybeBeforeSignature', text: 'DeclaredMethod'},
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
// CHECK:STDOUT:     {kind: 'NamedConstraintDefinition', text: '}', subtree_size: 17},
// CHECK:STDOUT:         {kind: 'NamedConstraintIntroducer', text: 'constraint'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeSignature', text: 'Foo'},
// CHECK:STDOUT:       {kind: 'NamedConstraintDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:           {kind: 'IdentifierNameMaybeBeforeSignature', text: 'DefinedMethod'},
// CHECK:STDOUT:             {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:                 {kind: 'SelfValueName', text: 'self'},
// CHECK:STDOUT:               {kind: 'SelfBindingPattern', text: 'self', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'UnusedPattern', text: 'unused', subtree_size: 3},
// CHECK:STDOUT:             {kind: 'PatternListComma', text: ','},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeSignature', text: 'b'},
// CHECK:STDOUT:               {kind: 'SelfTypeNameExpr', text: 'Self'},
// CHECK:STDOUT:             {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'ExplicitParamList', text: ')', subtree_size: 9},
// CHECK:STDOUT:             {kind: 'SelfTypeNameExpr', text: 'Self'},
// CHECK:STDOUT:           {kind: 'ReturnType', text: '->', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 14},
// CHECK:STDOUT:       {kind: 'FunctionDefinition', text: '}', subtree_size: 15},
// CHECK:STDOUT:     {kind: 'NamedConstraintDefinition', text: '}', subtree_size: 19},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
