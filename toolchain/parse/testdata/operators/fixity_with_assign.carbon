// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() {
  t = t**;
  n = n * n;
  n = n * *p;
  n = n*n;
}

// CHECK:STDOUT: - filename: fixity_with_assign.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'Name', text: 'F'},
// CHECK:STDOUT:           {kind: 'ParameterListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ParameterList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'NameExpression', text: 't'},
// CHECK:STDOUT:               {kind: 'NameExpression', text: 't'},
// CHECK:STDOUT:             {kind: 'PostfixOperator', text: '*', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'PostfixOperator', text: '*', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'InfixOperator', text: '=', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ExpressionStatement', text: ';', subtree_size: 6},
// CHECK:STDOUT:           {kind: 'NameExpression', text: 'n'},
// CHECK:STDOUT:             {kind: 'NameExpression', text: 'n'},
// CHECK:STDOUT:             {kind: 'NameExpression', text: 'n'},
// CHECK:STDOUT:           {kind: 'InfixOperator', text: '*', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'InfixOperator', text: '=', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ExpressionStatement', text: ';', subtree_size: 6},
// CHECK:STDOUT:           {kind: 'NameExpression', text: 'n'},
// CHECK:STDOUT:             {kind: 'NameExpression', text: 'n'},
// CHECK:STDOUT:               {kind: 'NameExpression', text: 'p'},
// CHECK:STDOUT:             {kind: 'PrefixOperator', text: '*', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'InfixOperator', text: '*', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'InfixOperator', text: '=', subtree_size: 6},
// CHECK:STDOUT:       {kind: 'ExpressionStatement', text: ';', subtree_size: 7},
// CHECK:STDOUT:           {kind: 'NameExpression', text: 'n'},
// CHECK:STDOUT:             {kind: 'NameExpression', text: 'n'},
// CHECK:STDOUT:             {kind: 'NameExpression', text: 'n'},
// CHECK:STDOUT:           {kind: 'InfixOperator', text: '*', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'InfixOperator', text: '=', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ExpressionStatement', text: ';', subtree_size: 6},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 31},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
