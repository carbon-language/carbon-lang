// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{carbon-run-parser}
// CHECK:STDOUT: [
// CHECK:STDOUT:     {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:     {kind: 'DeclaredName', text: 'F'},
// CHECK:STDOUT:       {kind: 'ParameterListStart', text: '('},
// CHECK:STDOUT:     {kind: 'ParameterList', text: ')', subtree_size: 2},
// CHECK:STDOUT:   {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'NameReference', text: 'n'},
// CHECK:STDOUT:             {kind: 'NameReference', text: 'a'},
// CHECK:STDOUT:             {kind: 'NameReference', text: 'b'},
// CHECK:STDOUT:           {kind: 'InfixOperator', text: '*', subtree_size: 3},
// CHECK:STDOUT:             {kind: 'NameReference', text: 'c'},
// CHECK:STDOUT:             {kind: 'NameReference', text: 'd'},
// CHECK:STDOUT:           {kind: 'InfixOperator', text: '*', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'InfixOperator', text: '+', subtree_size: 7},
// CHECK:STDOUT:                 {kind: 'NameReference', text: 'd'},
// CHECK:STDOUT:                 {kind: 'NameReference', text: 'd'},
// CHECK:STDOUT:               {kind: 'InfixOperator', text: '*', subtree_size: 3},
// CHECK:STDOUT:               {kind: 'NameReference', text: 'e'},
// CHECK:STDOUT:             {kind: 'InfixOperator', text: '<<', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:             {kind: 'NameReference', text: 'f'},
// CHECK:STDOUT:           {kind: 'InfixOperator', text: '&', has_error: yes, subtree_size: 7},
// CHECK:STDOUT:             {kind: 'NameReference', text: 'g'},
// CHECK:STDOUT:           {kind: 'PrefixOperator', text: 'not', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'InfixOperator', text: '-', has_error: yes, subtree_size: 10},
// CHECK:STDOUT:       {kind: 'InfixOperator', text: '=', subtree_size: 18},
// CHECK:STDOUT:     {kind: 'InfixOperator', text: '=', subtree_size: 20},
// CHECK:STDOUT:   {kind: 'ExpressionStatement', text: ';', subtree_size: 21},
// CHECK:STDOUT: {kind: 'FunctionDefinition', text: '}', subtree_size: 27},
// CHECK:STDOUT: {kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

fn F() {
  // CHECK:STDERR: {{.*}}/toolchain/parser/testdata/operators/fail_variety.carbon:[[@LINE+4]]:29: Parentheses are required to disambiguate operator precedence.
  // CHECK:STDERR: {{.*}}/toolchain/parser/testdata/operators/fail_variety.carbon:[[@LINE+3]]:34: Parentheses are required to disambiguate operator precedence.
  // CHECK:STDERR: {{.*}}/toolchain/parser/testdata/operators/fail_variety.carbon:[[@LINE+2]]:38: Parentheses are required to disambiguate operator precedence.
  // CHECK:STDERR: {{.*}}/toolchain/parser/testdata/operators/fail_variety.carbon:[[@LINE+1]]:40: Parentheses are required to disambiguate operator precedence.
  n = a * b + c * d = d * d << e & f - not g;
}
