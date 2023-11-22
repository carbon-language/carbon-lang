// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn foo(bar: i64, baz: i64) {
  foo(baz, bar + baz);
}

// CHECK:STDOUT: - filename: with_params.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'Name', text: 'foo'},
// CHECK:STDOUT:           {kind: 'ParamListStart', text: '('},
// CHECK:STDOUT:             {kind: 'Name', text: 'bar'},
// CHECK:STDOUT:             {kind: 'IntegerTypeLiteral', text: 'i64'},
// CHECK:STDOUT:           {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'ParamListComma', text: ','},
// CHECK:STDOUT:             {kind: 'Name', text: 'baz'},
// CHECK:STDOUT:             {kind: 'IntegerTypeLiteral', text: 'i64'},
// CHECK:STDOUT:           {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'ParamList', text: ')', subtree_size: 9},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 12},
// CHECK:STDOUT:             {kind: 'NameExpr', text: 'foo'},
// CHECK:STDOUT:           {kind: 'CallExprStart', text: '(', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'NameExpr', text: 'baz'},
// CHECK:STDOUT:           {kind: 'CallExprComma', text: ','},
// CHECK:STDOUT:             {kind: 'NameExpr', text: 'bar'},
// CHECK:STDOUT:             {kind: 'NameExpr', text: 'baz'},
// CHECK:STDOUT:           {kind: 'InfixOperator', text: '+', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'CallExpr', text: ')', subtree_size: 8},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 9},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 22},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
