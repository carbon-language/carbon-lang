// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F(p: i32*) {
  // Numeric, bitwise, and bit-shift binary operators are lower precedence than
  // Numeric and bitwise unary operators, which are lower precedence than
  // dereference.
  -*p + ^*p;
  -*p | ^*p;
  -*p << ^*p;
}

// CHECK:STDOUT: [
// CHECK:STDOUT:     {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:     {kind: 'Name', text: 'F'},
// CHECK:STDOUT:       {kind: 'ParameterListStart', text: '('},
// CHECK:STDOUT:         {kind: 'Name', text: 'p'},
// CHECK:STDOUT:           {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:         {kind: 'PostfixOperator', text: '*', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'PatternBinding', text: ':', subtree_size: 4},
// CHECK:STDOUT:     {kind: 'ParameterList', text: ')', subtree_size: 6},
// CHECK:STDOUT:   {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 9},
// CHECK:STDOUT:           {kind: 'NameExpression', text: 'p'},
// CHECK:STDOUT:         {kind: 'PrefixOperator', text: '*', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'PrefixOperator', text: '-', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'NameExpression', text: 'p'},
// CHECK:STDOUT:         {kind: 'PrefixOperator', text: '*', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'PrefixOperator', text: '^', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'InfixOperator', text: '+', subtree_size: 7},
// CHECK:STDOUT:   {kind: 'ExpressionStatement', text: ';', subtree_size: 8},
// CHECK:STDOUT:           {kind: 'NameExpression', text: 'p'},
// CHECK:STDOUT:         {kind: 'PrefixOperator', text: '*', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'PrefixOperator', text: '-', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'NameExpression', text: 'p'},
// CHECK:STDOUT:         {kind: 'PrefixOperator', text: '*', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'PrefixOperator', text: '^', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'InfixOperator', text: '|', subtree_size: 7},
// CHECK:STDOUT:   {kind: 'ExpressionStatement', text: ';', subtree_size: 8},
// CHECK:STDOUT:           {kind: 'NameExpression', text: 'p'},
// CHECK:STDOUT:         {kind: 'PrefixOperator', text: '*', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'PrefixOperator', text: '-', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'NameExpression', text: 'p'},
// CHECK:STDOUT:         {kind: 'PrefixOperator', text: '*', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'PrefixOperator', text: '^', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'InfixOperator', text: '<<', subtree_size: 7},
// CHECK:STDOUT:   {kind: 'ExpressionStatement', text: ';', subtree_size: 8},
// CHECK:STDOUT: {kind: 'FunctionDefinition', text: '}', subtree_size: 34},
// CHECK:STDOUT: {kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]
