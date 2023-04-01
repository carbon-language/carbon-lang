// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{carbon-run-parser}
// CHECK:STDOUT: [
// CHECK:STDOUT:   {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:     {kind: 'DeclaredName', text: 'n'},
// CHECK:STDOUT:     {kind: 'Literal', text: 'i8'},
// CHECK:STDOUT:   {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:   {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:       {kind: 'NameReference', text: 'n'},
// CHECK:STDOUT:     {kind: 'PostfixOperator', text: '*', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'NameReference', text: 'n'},
// CHECK:STDOUT:   {kind: 'InfixOperator', text: '-', has_error: yes, subtree_size: 4},
// CHECK:STDOUT: {kind: 'VariableDeclaration', text: ';', subtree_size: 10},
// CHECK:STDOUT: {kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

// CHECK:STDERR: {{.*}}/toolchain/parser/testdata/operators/fail_precedence_star_minus.carbon:[[@LINE+1]]:16: Parentheses are required to disambiguate operator precedence.
var n: i8 = n* -n;
