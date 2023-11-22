// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F(n: i32) {
  // Type operators and unary operators are lower precedence than `as`.
  -n as const i32;
  &n as i32*;

  // `as` is lower precedence than relational comparisons and logical operators.
  if (1 as i32 < 2 as i32 and true as bool and false as bool) {}
}

// CHECK:STDOUT: - filename: precedence_as.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'Name', text: 'F'},
// CHECK:STDOUT:           {kind: 'ParamListStart', text: '('},
// CHECK:STDOUT:             {kind: 'Name', text: 'n'},
// CHECK:STDOUT:             {kind: 'IntegerTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'ParamList', text: ')', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 8},
// CHECK:STDOUT:             {kind: 'NameExpr', text: 'n'},
// CHECK:STDOUT:           {kind: 'PrefixOperator', text: '-', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'IntegerTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'PrefixOperator', text: 'const', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'InfixOperator', text: 'as', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 6},
// CHECK:STDOUT:             {kind: 'NameExpr', text: 'n'},
// CHECK:STDOUT:           {kind: 'PrefixOperator', text: '&', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'IntegerTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'PostfixOperator', text: '*', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'InfixOperator', text: 'as', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 6},
// CHECK:STDOUT:           {kind: 'IfConditionStart', text: '('},
// CHECK:STDOUT:                       {kind: 'IntegerLiteral', text: '1'},
// CHECK:STDOUT:                       {kind: 'IntegerTypeLiteral', text: 'i32'},
// CHECK:STDOUT:                     {kind: 'InfixOperator', text: 'as', subtree_size: 3},
// CHECK:STDOUT:                       {kind: 'IntegerLiteral', text: '2'},
// CHECK:STDOUT:                       {kind: 'IntegerTypeLiteral', text: 'i32'},
// CHECK:STDOUT:                     {kind: 'InfixOperator', text: 'as', subtree_size: 3},
// CHECK:STDOUT:                   {kind: 'InfixOperator', text: '<', subtree_size: 7},
// CHECK:STDOUT:                 {kind: 'ShortCircuitOperand', text: 'and', subtree_size: 8},
// CHECK:STDOUT:                   {kind: 'BoolLiteralTrue', text: 'true'},
// CHECK:STDOUT:                   {kind: 'BoolTypeLiteral', text: 'bool'},
// CHECK:STDOUT:                 {kind: 'InfixOperator', text: 'as', subtree_size: 3},
// CHECK:STDOUT:               {kind: 'InfixOperator', text: 'and', subtree_size: 12},
// CHECK:STDOUT:             {kind: 'ShortCircuitOperand', text: 'and', subtree_size: 13},
// CHECK:STDOUT:               {kind: 'BoolLiteralFalse', text: 'false'},
// CHECK:STDOUT:               {kind: 'BoolTypeLiteral', text: 'bool'},
// CHECK:STDOUT:             {kind: 'InfixOperator', text: 'as', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'InfixOperator', text: 'and', subtree_size: 17},
// CHECK:STDOUT:         {kind: 'IfCondition', text: ')', subtree_size: 19},
// CHECK:STDOUT:           {kind: 'CodeBlockStart', text: '{'},
// CHECK:STDOUT:         {kind: 'CodeBlock', text: '}', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'IfStatement', text: 'if', subtree_size: 22},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 43},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
