// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{carbon-run-parser}
// CHECK:STDOUT: [
// CHECK:STDOUT:   {kind: 'DeclaredName', text: 'foo'},
// CHECK:STDOUT:     {kind: 'InterfaceBodyStart', text: '{'},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'DeclaredName', text: 'OnlyMe'},
// CHECK:STDOUT:         {kind: 'MeParamStart', text: '['},
// CHECK:STDOUT:           {kind: 'Me', text: 'me'},
// CHECK:STDOUT:           {kind: 'Self', text: 'Self'},
// CHECK:STDOUT:         {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'MeParam', text: ']', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'ParameterListStart', text: '('},
// CHECK:STDOUT:           {kind: 'DeclaredName', text: 'other'},
// CHECK:STDOUT:           {kind: 'Self', text: 'Self'},
// CHECK:STDOUT:         {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ParameterList', text: ')', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'Self', text: 'Self'},
// CHECK:STDOUT:       {kind: 'ReturnType', text: '->', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'FunctionDeclaration', text: ';', subtree_size: 15},
// CHECK:STDOUT:       {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:       {kind: 'DeclaredName', text: 'MeAddr'},
// CHECK:STDOUT:         {kind: 'MeParamStart', text: '['},
// CHECK:STDOUT:             {kind: 'Addr', text: 'addr'},
// CHECK:STDOUT:           {kind: 'MeAddr', text: 'me', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'Self', text: 'Self'},
// CHECK:STDOUT:         {kind: 'PatternBinding', text: ':', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'MeParam', text: ']', subtree_size: 6},
// CHECK:STDOUT:         {kind: 'ParameterListStart', text: '('},
// CHECK:STDOUT:           {kind: 'DeclaredName', text: 'other'},
// CHECK:STDOUT:           {kind: 'Self', text: 'Self'},
// CHECK:STDOUT:         {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ParameterList', text: ')', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'Self', text: 'Self'},
// CHECK:STDOUT:       {kind: 'ReturnType', text: '->', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'FunctionDeclaration', text: ';', subtree_size: 16},
// CHECK:STDOUT:   {kind: 'InterfaceBody', text: '}', subtree_size: 33},
// CHECK:STDOUT: {kind: 'InterfaceDefinition', text: 'interface', subtree_size: 35},
// CHECK:STDOUT: {kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]
interface foo {
  fn OnlyMe[me : Self](other: Self) -> Self;

  fn MeAddr[addr me : Self](other: Self) -> Self;
}
