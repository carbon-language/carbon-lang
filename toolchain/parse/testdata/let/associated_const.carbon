// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/let/associated_const.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/let/associated_const.carbon

interface MagicBox {
  let secretMessage:! String;

  // While parsing, we allow having default initializer for the associated constants.
  let notASecretMessage:! String = "This is not a secret message";
}

// CHECK:STDOUT: - filename: associated_const.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'MagicBox'},
// CHECK:STDOUT:       {kind: 'InterfaceDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'AssociatedConstantIntroducer', text: 'let'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'secretMessage'},
// CHECK:STDOUT:           {kind: 'StringTypeLiteral', text: 'String'},
// CHECK:STDOUT:         {kind: 'AssociatedConstantNameAndType', text: ':!', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'AssociatedConstantDecl', text: ';', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'AssociatedConstantIntroducer', text: 'let'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'notASecretMessage'},
// CHECK:STDOUT:           {kind: 'StringTypeLiteral', text: 'String'},
// CHECK:STDOUT:         {kind: 'AssociatedConstantNameAndType', text: ':!', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'AssociatedConstantInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'StringLiteral', text: '"This is not a secret message"'},
// CHECK:STDOUT:       {kind: 'AssociatedConstantDecl', text: ';', subtree_size: 7},
// CHECK:STDOUT:     {kind: 'InterfaceDefinition', text: '}', subtree_size: 16},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
