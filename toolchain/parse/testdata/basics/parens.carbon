// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F(n: i32) -> i32 {
  return ((((n))));
}

// CHECK:STDOUT: - filename: parens.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'Name', text: 'F'},
// CHECK:STDOUT:           {kind: 'ParameterListStart', text: '('},
// CHECK:STDOUT:             {kind: 'Name', text: 'n'},
// CHECK:STDOUT:             {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:           {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'ParameterList', text: ')', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:         {kind: 'ReturnType', text: '->', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 10},
// CHECK:STDOUT:         {kind: 'ReturnStatementStart', text: 'return'},
// CHECK:STDOUT:           {kind: 'ParenExpressionOrTupleLiteralStart', text: '('},
// CHECK:STDOUT:             {kind: 'ParenExpressionOrTupleLiteralStart', text: '('},
// CHECK:STDOUT:               {kind: 'ParenExpressionOrTupleLiteralStart', text: '('},
// CHECK:STDOUT:                 {kind: 'ParenExpressionOrTupleLiteralStart', text: '('},
// CHECK:STDOUT:                 {kind: 'NameExpression', text: 'n'},
// CHECK:STDOUT:               {kind: 'ParenExpression', text: ')', subtree_size: 3},
// CHECK:STDOUT:             {kind: 'ParenExpression', text: ')', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'ParenExpression', text: ')', subtree_size: 7},
// CHECK:STDOUT:         {kind: 'ParenExpression', text: ')', subtree_size: 9},
// CHECK:STDOUT:       {kind: 'ReturnStatement', text: ';', subtree_size: 11},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 22},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
