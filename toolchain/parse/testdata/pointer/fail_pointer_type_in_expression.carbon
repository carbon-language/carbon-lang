// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> i32 {
  // TODO: Indicate the locations of both operators involved in the precedence
  // error, so that it's clear what ambiguity we're referring to.
  // TODO: Improve error recovery so that we recover as `3 * (i32*) * 4`, not
  // as `(3 * i32)* * 4`, to suppress the second error here.
  // CHECK:STDERR: fail_pointer_type_in_expression.carbon:[[@LINE+6]]:17: ERROR: Parentheses are required to disambiguate operator precedence.
  // CHECK:STDERR:   return 3 * i32* * 4;
  // CHECK:STDERR:                 ^
  // CHECK:STDERR: fail_pointer_type_in_expression.carbon:[[@LINE+3]]:19: ERROR: Parentheses are required to disambiguate operator precedence.
  // CHECK:STDERR:   return 3 * i32* * 4;
  // CHECK:STDERR:                   ^
  return 3 * i32* * 4;
}

fn G() -> i32 {
  // CHECK:STDERR: fail_pointer_type_in_expression.carbon:[[@LINE+3]]:15: ERROR: Parentheses are required to disambiguate operator precedence.
  // CHECK:STDERR:   return i32* + 4;
  // CHECK:STDERR:               ^
  return i32* + 4;
}

fn H() -> i32 {
  // CHECK:STDERR: fail_pointer_type_in_expression.carbon:[[@LINE+3]]:14: ERROR: Parentheses are required to disambiguate operator precedence.
  // CHECK:STDERR:   return *i32*;
  // CHECK:STDERR:              ^
  return *i32*;
}

// CHECK:STDOUT: - filename: fail_pointer_type_in_expression.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'Name', text: 'F'},
// CHECK:STDOUT:           {kind: 'ParameterListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ParameterList', text: ')', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:         {kind: 'ReturnType', text: '->', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 7},
// CHECK:STDOUT:         {kind: 'ReturnStatementStart', text: 'return'},
// CHECK:STDOUT:               {kind: 'Literal', text: '3'},
// CHECK:STDOUT:               {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:             {kind: 'InfixOperator', text: '*', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'PostfixOperator', text: '*', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:           {kind: 'Literal', text: '4'},
// CHECK:STDOUT:         {kind: 'InfixOperator', text: '*', has_error: yes, subtree_size: 6},
// CHECK:STDOUT:       {kind: 'ReturnStatement', text: ';', subtree_size: 8},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 16},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'Name', text: 'G'},
// CHECK:STDOUT:           {kind: 'ParameterListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ParameterList', text: ')', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:         {kind: 'ReturnType', text: '->', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 7},
// CHECK:STDOUT:         {kind: 'ReturnStatementStart', text: 'return'},
// CHECK:STDOUT:             {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:           {kind: 'PostfixOperator', text: '*', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'Literal', text: '4'},
// CHECK:STDOUT:         {kind: 'InfixOperator', text: '+', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:       {kind: 'ReturnStatement', text: ';', subtree_size: 6},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 14},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'Name', text: 'H'},
// CHECK:STDOUT:           {kind: 'ParameterListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ParameterList', text: ')', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:         {kind: 'ReturnType', text: '->', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 7},
// CHECK:STDOUT:         {kind: 'ReturnStatementStart', text: 'return'},
// CHECK:STDOUT:             {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:           {kind: 'PrefixOperator', text: '*', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'PostfixOperator', text: '*', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ReturnStatement', text: ';', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 13},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
