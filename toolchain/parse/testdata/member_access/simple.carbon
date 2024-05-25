// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F(a: T) {
  a.x;
  a->x;
  a.x->y;
  a->x.y;
}

// CHECK:STDOUT: - filename: simple.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'F'},
// CHECK:STDOUT:           {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:             {kind: 'IdentifierName', text: 'a'},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'T'},
// CHECK:STDOUT:           {kind: 'BindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'TuplePattern', text: ')', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 8},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'a'},
// CHECK:STDOUT:           {kind: 'IdentifierName', text: 'x'},
// CHECK:STDOUT:         {kind: 'MemberAccessExpr', text: '.', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'a'},
// CHECK:STDOUT:           {kind: 'IdentifierName', text: 'x'},
// CHECK:STDOUT:         {kind: 'PointerMemberAccessExpr', text: '->', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 4},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'a'},
// CHECK:STDOUT:             {kind: 'IdentifierName', text: 'x'},
// CHECK:STDOUT:           {kind: 'MemberAccessExpr', text: '.', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'IdentifierName', text: 'y'},
// CHECK:STDOUT:         {kind: 'PointerMemberAccessExpr', text: '->', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 6},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'a'},
// CHECK:STDOUT:             {kind: 'IdentifierName', text: 'x'},
// CHECK:STDOUT:           {kind: 'PointerMemberAccessExpr', text: '->', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'IdentifierName', text: 'y'},
// CHECK:STDOUT:         {kind: 'MemberAccessExpr', text: '.', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 6},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 29},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
