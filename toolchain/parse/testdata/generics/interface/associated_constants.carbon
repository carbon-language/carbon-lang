// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/generics/interface/associated_constants.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/generics/interface/associated_constants.carbon

interface Foo {
  // TODO: support `let T:! type;`

  final let I: i32 = 4;
  default let D: bool = true;
}

// CHECK:STDOUT: - filename: associated_constants.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'Foo'},
// CHECK:STDOUT:       {kind: 'InterfaceDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'FinalModifier', text: 'final'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'I'},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'IntLiteral', text: '4'},
// CHECK:STDOUT:       {kind: 'LetDecl', text: ';', subtree_size: 8},
// CHECK:STDOUT:         {kind: 'LetIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'DefaultModifier', text: 'default'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'D'},
// CHECK:STDOUT:           {kind: 'BoolTypeLiteral', text: 'bool'},
// CHECK:STDOUT:         {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'LetInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'BoolLiteralTrue', text: 'true'},
// CHECK:STDOUT:       {kind: 'LetDecl', text: ';', subtree_size: 8},
// CHECK:STDOUT:     {kind: 'InterfaceDefinition', text: '}', subtree_size: 20},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
