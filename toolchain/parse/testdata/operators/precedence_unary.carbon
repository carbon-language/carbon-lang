// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F(p: i32*) {
  // Numeric, bitwise, and bit-shift binary operators are lower precedence than
  // numeric and bitwise unary operators, which are lower precedence than
  // dereference.
  -*p + ^*p;
  -*p | ^*p;
  -*p << ^*p;
}

// CHECK:STDOUT: - filename: precedence_unary.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'F'},
// CHECK:STDOUT:           {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:             {kind: 'IdentifierName', text: 'p'},
// CHECK:STDOUT:               {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:             {kind: 'PostfixOperator', text: '*', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'BindingPattern', text: ':', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'TuplePattern', text: ')', subtree_size: 6},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 9},
// CHECK:STDOUT:               {kind: 'IdentifierNameExpr', text: 'p'},
// CHECK:STDOUT:             {kind: 'PrefixOperator', text: '*', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'PrefixOperator', text: '-', subtree_size: 3},
// CHECK:STDOUT:               {kind: 'IdentifierNameExpr', text: 'p'},
// CHECK:STDOUT:             {kind: 'PrefixOperator', text: '*', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'PrefixOperator', text: '^', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'InfixOperator', text: '+', subtree_size: 7},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 8},
// CHECK:STDOUT:               {kind: 'IdentifierNameExpr', text: 'p'},
// CHECK:STDOUT:             {kind: 'PrefixOperator', text: '*', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'PrefixOperator', text: '-', subtree_size: 3},
// CHECK:STDOUT:               {kind: 'IdentifierNameExpr', text: 'p'},
// CHECK:STDOUT:             {kind: 'PrefixOperator', text: '*', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'PrefixOperator', text: '^', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'InfixOperator', text: '|', subtree_size: 7},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 8},
// CHECK:STDOUT:               {kind: 'IdentifierNameExpr', text: 'p'},
// CHECK:STDOUT:             {kind: 'PrefixOperator', text: '*', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'PrefixOperator', text: '-', subtree_size: 3},
// CHECK:STDOUT:               {kind: 'IdentifierNameExpr', text: 'p'},
// CHECK:STDOUT:             {kind: 'PrefixOperator', text: '*', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'PrefixOperator', text: '^', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'InfixOperator', text: '<<', subtree_size: 7},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 8},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 34},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
