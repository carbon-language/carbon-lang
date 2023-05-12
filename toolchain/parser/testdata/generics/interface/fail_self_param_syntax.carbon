// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: [
// CHECK:STDOUT:     {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:     {kind: 'DeclaredName', text: 'Foo'},
// CHECK:STDOUT:   {kind: 'InterfaceDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:     {kind: 'DeclaredName', text: 'Sub'},
// CHECK:STDOUT:       {kind: 'DeducedParameterListStart', text: '['},
// CHECK:STDOUT:         {kind: 'DeclaredName', text: 'me'},
// CHECK:STDOUT:         {kind: 'InvalidParse', text: 'Self', has_error: yes},
// CHECK:STDOUT:       {kind: 'PatternBinding', text: 'me', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'DeducedParameterList', text: ']', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ParameterListStart', text: '('},
// CHECK:STDOUT:         {kind: 'DeclaredName', text: 'b'},
// CHECK:STDOUT:         {kind: 'SelfTypeIdentifier', text: 'Self'},
// CHECK:STDOUT:       {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'ParameterList', text: ')', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'SelfTypeIdentifier', text: 'Self'},
// CHECK:STDOUT:     {kind: 'ReturnType', text: '->', subtree_size: 2},
// CHECK:STDOUT:   {kind: 'FunctionDeclaration', text: ';', subtree_size: 15},
// CHECK:STDOUT:     {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:     {kind: 'DeclaredName', text: 'Mul'},
// CHECK:STDOUT:       {kind: 'DeducedParameterListStart', text: '['},
// CHECK:STDOUT:         {kind: 'DeclaredName', text: 'Self', has_error: yes},
// CHECK:STDOUT:         {kind: 'InvalidParse', text: 'Self', has_error: yes},
// CHECK:STDOUT:       {kind: 'PatternBinding', text: 'Self', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'DeducedParameterList', text: ']', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ParameterListStart', text: '('},
// CHECK:STDOUT:         {kind: 'DeclaredName', text: 'b'},
// CHECK:STDOUT:         {kind: 'SelfTypeIdentifier', text: 'Self'},
// CHECK:STDOUT:       {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'ParameterList', text: ')', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'SelfTypeIdentifier', text: 'Self'},
// CHECK:STDOUT:     {kind: 'ReturnType', text: '->', subtree_size: 2},
// CHECK:STDOUT:   {kind: 'FunctionDeclaration', text: ';', subtree_size: 15},
// CHECK:STDOUT: {kind: 'InterfaceDefinition', text: '}', subtree_size: 34},
// CHECK:STDOUT: {kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

interface Foo {
  // CHECK:STDERR: toolchain/parser/testdata/generics/interface/fail_self_param_syntax.carbon:[[@LINE+1]]:13: Expected parameter declaration.
  fn Sub[me Self](b: Self) -> Self;

  // CHECK:STDERR: toolchain/parser/testdata/generics/interface/fail_self_param_syntax.carbon:[[@LINE+1]]:10: Expected parameter declaration.
  fn Mul[Self](b: Self) -> Self;
}
