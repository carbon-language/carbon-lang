// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/generics/interface/fail_associated_constants.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/generics/interface/fail_associated_constants.carbon

interface Foo {
  // CHECK:STDERR: fail_associated_constants.carbon:[[@LINE+4]]:16: error: expected `;`; associated constants cannot be initialized [ExpectedDeclSemi]
  // CHECK:STDERR:   let T:! type = i32;
  // CHECK:STDERR:                ^
  // CHECK:STDERR:
  let T:! type = i32;
  // CHECK:STDERR: fail_associated_constants.carbon:[[@LINE+4]]:21: error: expected `;`; associated constants cannot be initialized [ExpectedDeclSemi]
  // CHECK:STDERR:   final let I:! i32 = 4;
  // CHECK:STDERR:                     ^
  // CHECK:STDERR:
  final let I:! i32 = 4;
  // CHECK:STDERR: fail_associated_constants.carbon:[[@LINE+4]]:24: error: expected `;`; associated constants cannot be initialized [ExpectedDeclSemi]
  // CHECK:STDERR:   default let D:! bool = true;
  // CHECK:STDERR:                        ^
  // CHECK:STDERR:
  default let D:! bool = true;
}

// CHECK:STDOUT: - filename: fail_associated_constants.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'Foo'},
// CHECK:STDOUT:       {kind: 'InterfaceDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'AssociatedConstantIntroducer', text: 'let'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'T'},
// CHECK:STDOUT:           {kind: 'TypeTypeLiteral', text: 'type'},
// CHECK:STDOUT:         {kind: 'AssociatedConstantBindingPattern', text: ':!', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'AssociatedConstantDecl', text: ';', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'AssociatedConstantIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'FinalModifier', text: 'final'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'I'},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'AssociatedConstantBindingPattern', text: ':!', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'AssociatedConstantDecl', text: ';', subtree_size: 6},
// CHECK:STDOUT:         {kind: 'AssociatedConstantIntroducer', text: 'let'},
// CHECK:STDOUT:         {kind: 'DefaultModifier', text: 'default'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'D'},
// CHECK:STDOUT:           {kind: 'BoolTypeLiteral', text: 'bool'},
// CHECK:STDOUT:         {kind: 'AssociatedConstantBindingPattern', text: ':!', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'AssociatedConstantDecl', text: ';', subtree_size: 6},
// CHECK:STDOUT:     {kind: 'InterfaceDefinition', text: '}', subtree_size: 21},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
