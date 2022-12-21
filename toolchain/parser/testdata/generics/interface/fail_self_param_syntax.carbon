// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{carbon-run-parser}
// CHECK:STDOUT: [
// CHECK:STDOUT:   {kind: 'DeclaredName', text: 'foo'},
// CHECK:STDOUT:     {kind: 'InterfaceBodyStart', text: '{'},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'DeclaredName', text: 'Add'},
// CHECK:STDOUT:         {kind: 'ParameterListStart', text: '('},
// CHECK:STDOUT:           {kind: 'DeclaredName', text: 'b'},
// CHECK:STDOUT:           {kind: 'SelfType', text: 'Self'},
// CHECK:STDOUT:         {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ParameterList', text: ')', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'SelfType', text: 'Self'},
// CHECK:STDOUT:       {kind: 'ReturnType', text: '->', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'FunctionDeclaration', text: ';', subtree_size: 10},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'DeclaredName', text: 'Sub'},
// CHECK:STDOUT:         {kind: 'DeducedParameterListStart', text: '['},
// CHECK:STDOUT:       {kind: 'DeducedParameterList', text: ']', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:         {kind: 'ParameterListStart', text: '('},
// CHECK:STDOUT:           {kind: 'DeclaredName', text: 'b'},
// CHECK:STDOUT:           {kind: 'SelfType', text: 'Self'},
// CHECK:STDOUT:         {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ParameterList', text: ')', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'SelfType', text: 'Self'},
// CHECK:STDOUT:       {kind: 'ReturnType', text: '->', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'FunctionDeclaration', text: ';', subtree_size: 12},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'DeclaredName', text: 'Mul'},
// CHECK:STDOUT:         {kind: 'DeducedParameterListStart', text: '['},
// CHECK:STDOUT:       {kind: 'DeducedParameterList', text: ']', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:         {kind: 'ParameterListStart', text: '('},
// CHECK:STDOUT:           {kind: 'DeclaredName', text: 'b'},
// CHECK:STDOUT:           {kind: 'SelfType', text: 'Self'},
// CHECK:STDOUT:         {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ParameterList', text: ')', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'SelfType', text: 'Self'},
// CHECK:STDOUT:       {kind: 'ReturnType', text: '->', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'FunctionDeclaration', text: ';', subtree_size: 12},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'DeclaredName', text: 'Rem'},
// CHECK:STDOUT:         {kind: 'DeducedParameterListStart', text: '['},
// CHECK:STDOUT:       {kind: 'DeducedParameterList', text: ']', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'ParameterListStart', text: '('},
// CHECK:STDOUT:           {kind: 'DeclaredName', text: 'b'},
// CHECK:STDOUT:           {kind: 'SelfType', text: 'Self'},
// CHECK:STDOUT:         {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ParameterList', text: ')', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'SelfType', text: 'Self'},
// CHECK:STDOUT:       {kind: 'ReturnType', text: '->', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'FunctionDeclaration', text: ';', subtree_size: 12},
// CHECK:STDOUT:   {kind: 'InterfaceBody', text: '}', subtree_size: 48},
// CHECK:STDOUT: {kind: 'InterfaceDefinition', text: 'interface', subtree_size: 50},
// CHECK:STDOUT: {kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]
interface foo {
  fn Add(b: Self) -> Self;

  // CHECK:STDERR: {{.*}}/toolchain/parser/testdata/generics/interface/fail_self_param_syntax.carbon:[[@LINE+1]]:10: Deduced parameters must be of the form: `<name>: <Type>` or `addr <name>: <Type>`.
  fn Sub[me Self](b: Self) -> Self;

  // CHECK:STDERR: {{.*}}/toolchain/parser/testdata/generics/interface/fail_self_param_syntax.carbon:[[@LINE+1]]:10: Deduced parameters must be of the form: `<name>: <Type>` or `addr <name>: <Type>`.
  fn Mul[Self](b: Self) -> Self;

  // TODO: The recovery token is currently inserted after the `;`:
  // ```
  // fn Div[me: Self(b: Self) -> Self;]
  // ```
  // Fix and uncomment this to test error handling.
  // fn Div[me: Self(b: Self) -> Self;

  // It is fine to have an empty deduced parameter list.
  fn Rem[](b: Self) -> Self;
}
