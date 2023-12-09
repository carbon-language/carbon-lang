// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F(n: i32) {
  // No ordering between `not` and `as`.
  // CHECK:STDERR: fail_precedence_as.carbon:[[@LINE+3]]:12: ERROR: Parentheses are required to disambiguate operator precedence.
  // CHECK:STDERR:   not true as bool;
  // CHECK:STDERR:            ^~
  not true as bool;

  // No ordering between most binary operators and `as`.
  // CHECK:STDERR: fail_precedence_as.carbon:[[@LINE+3]]:9: ERROR: Parentheses are required to disambiguate operator precedence.
  // CHECK:STDERR:   1 + 1 as i32;
  // CHECK:STDERR:         ^~
  1 + 1 as i32;
  // CHECK:STDERR: fail_precedence_as.carbon:[[@LINE+3]]:9: ERROR: Parentheses are required to disambiguate operator precedence.
  // CHECK:STDERR:   5 % 2 as i32;
  // CHECK:STDERR:         ^~
  5 % 2 as i32;
}

// CHECK:STDOUT: - filename: fail_precedence_as.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'F'},
// CHECK:STDOUT:           {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:             {kind: 'IdentifierName', text: 'n'},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'BindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'TuplePattern', text: ')', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 8},
// CHECK:STDOUT:             {kind: 'BoolLiteralTrue', text: 'true'},
// CHECK:STDOUT:           {kind: 'PrefixOperator', text: 'not', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'BoolTypeLiteral', text: 'bool'},
// CHECK:STDOUT:         {kind: 'InfixOperator', text: 'as', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 5},
// CHECK:STDOUT:             {kind: 'IntLiteral', text: '1'},
// CHECK:STDOUT:             {kind: 'IntLiteral', text: '1'},
// CHECK:STDOUT:           {kind: 'InfixOperator', text: '+', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'InfixOperator', text: 'as', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 6},
// CHECK:STDOUT:             {kind: 'IntLiteral', text: '5'},
// CHECK:STDOUT:             {kind: 'IntLiteral', text: '2'},
// CHECK:STDOUT:           {kind: 'InfixOperator', text: '%', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'InfixOperator', text: 'as', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 6},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 26},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
