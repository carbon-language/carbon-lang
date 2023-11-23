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
// CHECK:STDOUT:           {kind: 'ParamListStart', text: '('},
// CHECK:STDOUT:             {kind: 'Name', text: 'n'},
// CHECK:STDOUT:             {kind: 'IntegerTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'ParamList', text: ')', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'IntegerTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'ReturnType', text: '->', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 10},
// CHECK:STDOUT:         {kind: 'ReturnStatementStart', text: 'return'},
// CHECK:STDOUT:           {kind: 'ParenExprOrTupleLiteralStart', text: '('},
// CHECK:STDOUT:             {kind: 'ParenExprOrTupleLiteralStart', text: '('},
// CHECK:STDOUT:               {kind: 'ParenExprOrTupleLiteralStart', text: '('},
// CHECK:STDOUT:                 {kind: 'ParenExprOrTupleLiteralStart', text: '('},
// CHECK:STDOUT:                 {kind: 'NameExpr', text: 'n'},
// CHECK:STDOUT:               {kind: 'ParenExpr', text: ')', subtree_size: 3},
// CHECK:STDOUT:             {kind: 'ParenExpr', text: ')', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'ParenExpr', text: ')', subtree_size: 7},
// CHECK:STDOUT:         {kind: 'ParenExpr', text: ')', subtree_size: 9},
// CHECK:STDOUT:       {kind: 'ReturnStatement', text: ';', subtree_size: 11},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 22},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
