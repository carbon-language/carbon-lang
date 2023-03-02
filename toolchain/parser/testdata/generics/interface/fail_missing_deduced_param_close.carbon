// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{carbon-run-parser}
// CHECK:STDOUT: [
// CHECK:STDOUT:     {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:     {kind: 'DeclaredName', text: 'Foo'},
// CHECK:STDOUT:   {kind: 'InterfaceDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:     {kind: 'DeclaredName', text: 'Div'},
// CHECK:STDOUT:       {kind: 'DeducedParameterListStart', text: '['},
// CHECK:STDOUT:         {kind: 'DeclaredName', text: 'me'},
// CHECK:STDOUT:             {kind: 'SelfType', text: 'Self'},
// CHECK:STDOUT:           {kind: 'CallExpressionStart', text: '(', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'NameReference', text: 'b'},
// CHECK:STDOUT:         {kind: 'CallExpression', text: ')', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:       {kind: 'PatternBinding', text: ':', subtree_size: 6},
// CHECK:STDOUT:     {kind: 'DeducedParameterList', text: ']', has_error: yes, subtree_size: 8},
// CHECK:STDOUT:   {kind: 'FunctionDeclaration', text: 'fn', has_error: yes, subtree_size: 11},
// CHECK:STDOUT: {kind: 'InterfaceDefinition', text: '}', subtree_size: 15},
// CHECK:STDOUT: {kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

interface Foo {
  // Fix and uncomment this to test error handling.
  // CHECK:STDERR: {{.*}}/toolchain/parser/testdata/generics/interface/fail_missing_deduced_param_close.carbon:[[@LINE+3]]:9: Closing symbol does not match most recent opening symbol.
  // CHECK:STDERR: {{.*}}/toolchain/parser/testdata/generics/interface/fail_missing_deduced_param_close.carbon:[[@LINE+2]]:20: Expected `,` or `)`.
  // CHECK:STDERR: {{.*}}/toolchain/parser/testdata/generics/interface/fail_missing_deduced_param_close.carbon:[[@LINE+1]]:28: Expected `,` or `]`.
  fn Div[me: Self(b: Self) -> Self;
// CHECK:STDERR: {{.*}}/toolchain/parser/testdata/generics/interface/fail_missing_deduced_param_close.carbon:[[@LINE+1]]:1: Expected `(` after function name.
}
