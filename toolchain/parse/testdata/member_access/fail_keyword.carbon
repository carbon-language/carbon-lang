// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() {
  // CHECK:STDERR: fail_keyword.carbon:[[@LINE+4]]:5: ERROR: Expected identifier after `.`.
  // CHECK:STDERR:   a.self;
  // CHECK:STDERR:     ^~~~
  // CHECK:STDERR:
  a.self;
  // CHECK:STDERR: fail_keyword.carbon:[[@LINE+4]]:5: ERROR: Expected identifier after `.`.
  // CHECK:STDERR:   a.Self;
  // CHECK:STDERR:     ^~~~
  // CHECK:STDERR:
  a.Self;
  // CHECK:STDERR: fail_keyword.carbon:[[@LINE+4]]:5: ERROR: Expected identifier after `.`.
  // CHECK:STDERR:   a.for;
  // CHECK:STDERR:     ^~~
  // CHECK:STDERR:
  a.for;
  // CHECK:STDERR: fail_keyword.carbon:[[@LINE+4]]:6: ERROR: Expected identifier after `->`.
  // CHECK:STDERR:   p->self;
  // CHECK:STDERR:      ^~~~
  // CHECK:STDERR:
  p->self;
  // CHECK:STDERR: fail_keyword.carbon:[[@LINE+4]]:6: ERROR: Expected identifier after `->`.
  // CHECK:STDERR:   p->Self;
  // CHECK:STDERR:      ^~~~
  // CHECK:STDERR:
  p->Self;
  // CHECK:STDERR: fail_keyword.carbon:[[@LINE+3]]:6: ERROR: Expected identifier after `->`.
  // CHECK:STDERR:   p->while;
  // CHECK:STDERR:      ^~~~~
  p->while;
}

// CHECK:STDOUT: - filename: fail_keyword.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'F'},
// CHECK:STDOUT:           {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:         {kind: 'TuplePattern', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'a'},
// CHECK:STDOUT:           {kind: 'IdentifierName', text: 'self', has_error: yes},
// CHECK:STDOUT:         {kind: 'MemberAccessExpr', text: '.', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'a'},
// CHECK:STDOUT:           {kind: 'IdentifierName', text: 'Self', has_error: yes},
// CHECK:STDOUT:         {kind: 'MemberAccessExpr', text: '.', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'a'},
// CHECK:STDOUT:           {kind: 'IdentifierName', text: 'for', has_error: yes},
// CHECK:STDOUT:         {kind: 'MemberAccessExpr', text: '.', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'p'},
// CHECK:STDOUT:           {kind: 'IdentifierName', text: 'self', has_error: yes},
// CHECK:STDOUT:         {kind: 'PointerMemberAccessExpr', text: '->', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'p'},
// CHECK:STDOUT:           {kind: 'IdentifierName', text: 'Self', has_error: yes},
// CHECK:STDOUT:         {kind: 'PointerMemberAccessExpr', text: '->', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'p'},
// CHECK:STDOUT:           {kind: 'IdentifierName', text: 'while', has_error: yes},
// CHECK:STDOUT:         {kind: 'PointerMemberAccessExpr', text: '->', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 4},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 30},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
