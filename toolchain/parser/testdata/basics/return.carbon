// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{carbon-run-parser}
// CHECK:STDOUT: [
// CHECK:STDOUT:     {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:     {kind: 'DeclaredName', text: 'F'},
// CHECK:STDOUT:       {kind: 'ParameterListEnd', text: ')'},
// CHECK:STDOUT:     {kind: 'ParameterList', text: '(', subtree_size: 2},
// CHECK:STDOUT:   {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'NameReference', text: 'c'},
// CHECK:STDOUT:       {kind: 'ConditionEnd', text: ')'},
// CHECK:STDOUT:     {kind: 'Condition', text: '(', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'CodeBlockStart', text: '{'},
// CHECK:STDOUT:         {kind: 'StatementEnd', text: ';'},
// CHECK:STDOUT:       {kind: 'ReturnStatement', text: 'return', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'CodeBlock', text: '}', subtree_size: 4},
// CHECK:STDOUT:   {kind: 'IfStatement', text: 'if', subtree_size: 8},
// CHECK:STDOUT: {kind: 'FunctionDefinition', text: '}', subtree_size: 14},
// CHECK:STDOUT:     {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:     {kind: 'DeclaredName', text: 'G'},
// CHECK:STDOUT:         {kind: 'DeclaredName', text: 'x'},
// CHECK:STDOUT:         {kind: 'NameReference', text: 'Foo'},
// CHECK:STDOUT:       {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ParameterListEnd', text: ')'},
// CHECK:STDOUT:     {kind: 'ParameterList', text: '(', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'NameReference', text: 'Foo'},
// CHECK:STDOUT:     {kind: 'ReturnType', text: '->', subtree_size: 2},
// CHECK:STDOUT:   {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 10},
// CHECK:STDOUT:     {kind: 'NameReference', text: 'x'},
// CHECK:STDOUT:     {kind: 'StatementEnd', text: ';'},
// CHECK:STDOUT:   {kind: 'ReturnStatement', text: 'return', subtree_size: 3},
// CHECK:STDOUT: {kind: 'FunctionDefinition', text: '}', subtree_size: 14},
// CHECK:STDOUT: {kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

// NOTE: Move to its own directory when more tests are added.
fn F() {
  if (c) {
    return;
  }
}
fn G(x: Foo) -> Foo {
  return x;
}
