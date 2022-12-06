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
// CHECK:STDOUT:         {kind: 'DeclaredName', text: 'p'},
// CHECK:STDOUT:           {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:         {kind: 'PostfixOperator', text: '*', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'PatternBinding', text: ':', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'ParameterListComma', text: ','},
// CHECK:STDOUT:         {kind: 'DeclaredName', text: 'n'},
// CHECK:STDOUT:         {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:       {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'ParameterList', text: ')', subtree_size: 10},
// CHECK:STDOUT:   {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 13},
// CHECK:STDOUT: {kind: 'FunctionDefinition', text: '}', subtree_size: 14},
// CHECK:STDOUT: {kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

fn F(p: i32*, n: i32) {
}
