// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() {
  G(i32*, n * n);
}

// CHECK:STDOUT: - filename: fixity_in_call.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'Name', text: 'F'},
// CHECK:STDOUT:           {kind: 'ParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:             {kind: 'NameExpr', text: 'G'},
// CHECK:STDOUT:           {kind: 'CallExprStart', text: '(', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'IntegerTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'PostfixOperator', text: '*', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'CallExprComma', text: ','},
// CHECK:STDOUT:             {kind: 'NameExpr', text: 'n'},
// CHECK:STDOUT:             {kind: 'NameExpr', text: 'n'},
// CHECK:STDOUT:           {kind: 'InfixOperator', text: '*', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'CallExpr', text: ')', subtree_size: 9},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 10},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 16},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
