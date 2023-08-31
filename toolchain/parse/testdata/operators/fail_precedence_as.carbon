// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F(n: i32) {
  // No ordering between `not` and `as`.
  // CHECK:STDERR: fail_precedence_as.carbon:[[@LINE+3]]:12: Parentheses are required to disambiguate operator precedence.
  // CHECK:STDERR:   not true as bool;
  // CHECK:STDERR:            ^
  not true as bool;

  // No ordering between most binary operators and `as`.
  // CHECK:STDERR: fail_precedence_as.carbon:[[@LINE+3]]:9: Parentheses are required to disambiguate operator precedence.
  // CHECK:STDERR:   1 + 1 as i32;
  // CHECK:STDERR:         ^
  1 + 1 as i32;
  // CHECK:STDERR: fail_precedence_as.carbon:[[@LINE+3]]:9: Parentheses are required to disambiguate operator precedence.
  // CHECK:STDERR:   5 % 2 as i32;
  // CHECK:STDERR:         ^
  5 % 2 as i32;
}

// CHECK:STDOUT: [
// CHECK:STDOUT:     {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:     {kind: 'Name', text: 'F'},
// CHECK:STDOUT:       {kind: 'ParameterListStart', text: '('},
// CHECK:STDOUT:         {kind: 'Name', text: 'n'},
// CHECK:STDOUT:         {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:       {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'ParameterList', text: ')', subtree_size: 5},
// CHECK:STDOUT:   {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 8},
// CHECK:STDOUT:         {kind: 'Literal', text: 'true'},
// CHECK:STDOUT:       {kind: 'PrefixOperator', text: 'not', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'Literal', text: 'bool'},
// CHECK:STDOUT:     {kind: 'InfixOperator', text: 'as', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:   {kind: 'ExpressionStatement', text: ';', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'Literal', text: '1'},
// CHECK:STDOUT:         {kind: 'Literal', text: '1'},
// CHECK:STDOUT:       {kind: 'InfixOperator', text: '+', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:     {kind: 'InfixOperator', text: 'as', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:   {kind: 'ExpressionStatement', text: ';', subtree_size: 6},
// CHECK:STDOUT:         {kind: 'Literal', text: '5'},
// CHECK:STDOUT:         {kind: 'Literal', text: '2'},
// CHECK:STDOUT:       {kind: 'InfixOperator', text: '%', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:     {kind: 'InfixOperator', text: 'as', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:   {kind: 'ExpressionStatement', text: ';', subtree_size: 6},
// CHECK:STDOUT: {kind: 'FunctionDefinition', text: '}', subtree_size: 26},
// CHECK:STDOUT: {kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]
