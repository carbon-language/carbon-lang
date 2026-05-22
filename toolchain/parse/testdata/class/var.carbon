// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/class/var.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/class/var.carbon

class Foo {
  var x: i32;
  var y: i32 = 0;
  static var z: i32;
}


// CHECK:STDOUT: - filename: var.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'ClassIntroducer', text: 'class'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeSignature', text: 'Foo'},
// CHECK:STDOUT:       {kind: 'ClassDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'VarBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'VariablePattern', text: 'var', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'VariableDecl', text: ';', subtree_size: 6},
// CHECK:STDOUT:         {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'y'},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'VarBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'VariablePattern', text: 'var', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'IntLiteral', text: '0'},
// CHECK:STDOUT:       {kind: 'VariableDecl', text: ';', subtree_size: 8},
// CHECK:STDOUT:         {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:         {kind: 'StaticModifier', text: 'static'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeSignature', text: 'z'},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'VarBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'VariablePattern', text: 'var', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'VariableDecl', text: ';', subtree_size: 7},
// CHECK:STDOUT:     {kind: 'ClassDefinition', text: '}', subtree_size: 25},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
