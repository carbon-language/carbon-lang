// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// NOTE: Move to its own directory when more tests are added.
fn F() {
  a.b.f(c.d, (e)).g();
}

// CHECK:STDOUT: - filename: function_call.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'Name', text: 'F'},
// CHECK:STDOUT:           {kind: 'ParameterListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ParameterList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:                       {kind: 'NameExpr', text: 'a'},
// CHECK:STDOUT:                       {kind: 'Name', text: 'b'},
// CHECK:STDOUT:                     {kind: 'MemberAccessExpr', text: '.', subtree_size: 3},
// CHECK:STDOUT:                     {kind: 'Name', text: 'f'},
// CHECK:STDOUT:                   {kind: 'MemberAccessExpr', text: '.', subtree_size: 5},
// CHECK:STDOUT:                 {kind: 'CallExprStart', text: '(', subtree_size: 6},
// CHECK:STDOUT:                   {kind: 'NameExpr', text: 'c'},
// CHECK:STDOUT:                   {kind: 'Name', text: 'd'},
// CHECK:STDOUT:                 {kind: 'MemberAccessExpr', text: '.', subtree_size: 3},
// CHECK:STDOUT:                 {kind: 'CallExprComma', text: ','},
// CHECK:STDOUT:                   {kind: 'ParenExprOrTupleLiteralStart', text: '('},
// CHECK:STDOUT:                   {kind: 'NameExpr', text: 'e'},
// CHECK:STDOUT:                 {kind: 'ParenExpr', text: ')', subtree_size: 3},
// CHECK:STDOUT:               {kind: 'CallExpr', text: ')', subtree_size: 14},
// CHECK:STDOUT:               {kind: 'Name', text: 'g'},
// CHECK:STDOUT:             {kind: 'MemberAccessExpr', text: '.', subtree_size: 16},
// CHECK:STDOUT:           {kind: 'CallExprStart', text: '(', subtree_size: 17},
// CHECK:STDOUT:         {kind: 'CallExpr', text: ')', subtree_size: 18},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 19},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 25},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
