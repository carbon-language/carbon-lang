// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{carbon-run-parser}
// CHECK:STDOUT: [
// CHECK:STDOUT:   {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:     {kind: 'DeclaredName', text: 'v'},
// CHECK:STDOUT:     {kind: 'NameReference', text: 'Type'},
// CHECK:STDOUT:   {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'Literal', text: 'i8'},
// CHECK:STDOUT:     {kind: 'PostfixOperator', text: '*', subtree_size: 2},
// CHECK:STDOUT:   {kind: 'VariableInitializer', text: '=', subtree_size: 3},
// CHECK:STDOUT: {kind: 'VariableDeclaration', text: ';', subtree_size: 8},
// CHECK:STDOUT: {kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

var v: Type = i8*;
