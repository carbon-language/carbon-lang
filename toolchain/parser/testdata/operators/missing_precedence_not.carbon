// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{carbon-run-parser}
// CHECK:STDOUT: [
// CHECK:STDOUT:     {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:     {kind: 'DeclaredName', text: 'F'},
// CHECK:STDOUT:       {kind: 'ParameterListStart', text: '('},
// CHECK:STDOUT:     {kind: 'ParameterList', text: ')', subtree_size: 2},
// CHECK:STDOUT:   {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'NameReference', text: 'a'},
// CHECK:STDOUT:         {kind: 'PrefixOperator', text: 'not', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'NameReference', text: 'b'},
// CHECK:STDOUT:         {kind: 'PrefixOperator', text: 'not', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'InfixOperator', text: 'and', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'NameReference', text: 'c'},
// CHECK:STDOUT:       {kind: 'PrefixOperator', text: 'not', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'InfixOperator', text: 'and', subtree_size: 8},
// CHECK:STDOUT:   {kind: 'ExpressionStatement', text: ';', subtree_size: 9},
// CHECK:STDOUT: {kind: 'FunctionDefinition', text: '}', subtree_size: 15},
// CHECK:STDOUT: {kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

fn F() {
  // CHECK:STDERR: {{.*}}/toolchain/parser/testdata/operators/missing_precedence_not.carbon:[[@LINE+3]]:3: Whitespace is required before this unary operator.
  // CHECK:STDERR: {{.*}}/toolchain/parser/testdata/operators/missing_precedence_not.carbon:[[@LINE+2]]:13: Whitespace is required before this unary operator.
  // CHECK:STDERR: {{.*}}/toolchain/parser/testdata/operators/missing_precedence_not.carbon:[[@LINE+1]]:23: Whitespace is required before this unary operator.
  not a and not b and not c;
}
