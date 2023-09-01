// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() {
  // CHECK:STDERR: fail_variety.carbon:[[@LINE+15]]:21: Operator `=` can only be used as a complete statement.
  // CHECK:STDERR:   n = a * b + c * d = d * d << e & f - not g;
  // CHECK:STDERR:                     ^
  // CHECK:STDERR: fail_variety.carbon:[[@LINE+12]]:29: Parentheses are required to disambiguate operator precedence.
  // CHECK:STDERR:   n = a * b + c * d = d * d << e & f - not g;
  // CHECK:STDERR:                             ^
  // CHECK:STDERR: fail_variety.carbon:[[@LINE+9]]:34: Parentheses are required to disambiguate operator precedence.
  // CHECK:STDERR:   n = a * b + c * d = d * d << e & f - not g;
  // CHECK:STDERR:                                  ^
  // CHECK:STDERR: fail_variety.carbon:[[@LINE+6]]:38: Parentheses are required to disambiguate operator precedence.
  // CHECK:STDERR:   n = a * b + c * d = d * d << e & f - not g;
  // CHECK:STDERR:                                      ^
  // CHECK:STDERR: fail_variety.carbon:[[@LINE+3]]:40: Parentheses are required around this unary `not` operator.
  // CHECK:STDERR:   n = a * b + c * d = d * d << e & f - not g;
  // CHECK:STDERR:                                        ^
  n = a * b + c * d = d * d << e & f - not g;
}

// CHECK:STDOUT: [
// CHECK:STDOUT:     {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:     {kind: 'Name', text: 'F'},
// CHECK:STDOUT:       {kind: 'ParameterListStart', text: '('},
// CHECK:STDOUT:     {kind: 'ParameterList', text: ')', subtree_size: 2},
// CHECK:STDOUT:   {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'NameExpression', text: 'n'},
// CHECK:STDOUT:             {kind: 'NameExpression', text: 'a'},
// CHECK:STDOUT:             {kind: 'NameExpression', text: 'b'},
// CHECK:STDOUT:           {kind: 'InfixOperator', text: '*', subtree_size: 3},
// CHECK:STDOUT:             {kind: 'NameExpression', text: 'c'},
// CHECK:STDOUT:             {kind: 'NameExpression', text: 'd'},
// CHECK:STDOUT:           {kind: 'InfixOperator', text: '*', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'InfixOperator', text: '+', subtree_size: 7},
// CHECK:STDOUT:       {kind: 'InfixOperator', text: '=', subtree_size: 9},
// CHECK:STDOUT:               {kind: 'NameExpression', text: 'd'},
// CHECK:STDOUT:               {kind: 'NameExpression', text: 'd'},
// CHECK:STDOUT:             {kind: 'InfixOperator', text: '*', subtree_size: 3},
// CHECK:STDOUT:             {kind: 'NameExpression', text: 'e'},
// CHECK:STDOUT:           {kind: 'InfixOperator', text: '<<', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:           {kind: 'NameExpression', text: 'f'},
// CHECK:STDOUT:         {kind: 'InfixOperator', text: '&', has_error: yes, subtree_size: 7},
// CHECK:STDOUT:           {kind: 'NameExpression', text: 'g'},
// CHECK:STDOUT:         {kind: 'PrefixOperator', text: 'not', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'InfixOperator', text: '-', has_error: yes, subtree_size: 10},
// CHECK:STDOUT:     {kind: 'InfixOperator', text: '=', has_error: yes, subtree_size: 20},
// CHECK:STDOUT:   {kind: 'ExpressionStatement', text: ';', subtree_size: 21},
// CHECK:STDOUT: {kind: 'FunctionDefinition', text: '}', subtree_size: 27},
// CHECK:STDOUT: {kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]
