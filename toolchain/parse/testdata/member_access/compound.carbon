// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() {
  // CHECK:STDERR: compound.carbon:[[@LINE+3]]:5: ERROR: Expected identifier after `.`.
  // CHECK:STDERR:   a.(b);
  // CHECK:STDERR:     ^
  a.(b);
  // CHECK:STDERR: compound.carbon:[[@LINE+3]]:6: ERROR: Expected identifier after `->`.
  // CHECK:STDERR:   a->(b);
  // CHECK:STDERR:      ^
  a->(b);
}

// CHECK:STDOUT: - filename: compound.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'F'},
// CHECK:STDOUT:           {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:         {kind: 'TuplePattern', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:               {kind: 'IdentifierNameExpr', text: 'a'},
// CHECK:STDOUT:               {kind: 'IdentifierName', text: '(', has_error: yes},
// CHECK:STDOUT:             {kind: 'MemberAccessExpr', text: '.', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'CallExprStart', text: '(', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'b'},
// CHECK:STDOUT:         {kind: 'CallExpr', text: ')', has_error: yes, subtree_size: 6},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', has_error: yes, subtree_size: 7},
// CHECK:STDOUT:               {kind: 'IdentifierNameExpr', text: 'a'},
// CHECK:STDOUT:               {kind: 'IdentifierName', text: '(', has_error: yes},
// CHECK:STDOUT:             {kind: 'PointerMemberAccessExpr', text: '->', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'CallExprStart', text: '(', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'b'},
// CHECK:STDOUT:         {kind: 'CallExpr', text: ')', has_error: yes, subtree_size: 6},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', has_error: yes, subtree_size: 7},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 20},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
