// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: [
// CHECK:STDOUT:     {kind: 'ClassIntroducer', text: 'class'},
// CHECK:STDOUT:     {kind: 'DeclaredName', text: 'Foo'},
// CHECK:STDOUT:   {kind: 'ClassDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:     {kind: 'DeclaredName', text: 'Baz'},
// CHECK:STDOUT:       {kind: 'ParameterListStart', text: '('},
// CHECK:STDOUT:     {kind: 'ParameterList', text: ')', subtree_size: 2},
// CHECK:STDOUT:   {kind: 'FunctionDeclaration', text: ';', subtree_size: 5},
// CHECK:STDOUT: {kind: 'ClassDefinition', text: '}', subtree_size: 9},
// CHECK:STDOUT: {kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

class Foo {
  fn Baz();
}
