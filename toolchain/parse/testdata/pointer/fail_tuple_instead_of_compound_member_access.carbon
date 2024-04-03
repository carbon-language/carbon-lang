// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() {
  // CHECK:STDERR: fail_tuple_instead_of_compound_member_access.carbon:[[@LINE+4]]:6: ERROR: Expected expression.
  // CHECK:STDERR:   x.();
  // CHECK:STDERR:      ^
  // CHECK:STDERR:
  x.();

  // CHECK:STDERR: fail_tuple_instead_of_compound_member_access.carbon:[[@LINE+4]]:7: ERROR: Expected `)`.
  // CHECK:STDERR:   y.(1,);
  // CHECK:STDERR:       ^
  // CHECK:STDERR:
  y.(1,);

  // CHECK:STDERR: fail_tuple_instead_of_compound_member_access.carbon:[[@LINE+4]]:7: ERROR: Expected `)`.
  // CHECK:STDERR:   z.(a, b);
  // CHECK:STDERR:       ^
  // CHECK:STDERR:
  z.(a, b);

  // CHECK:STDERR: fail_tuple_instead_of_compound_member_access.carbon:[[@LINE+4]]:8: ERROR: Expected `)`.
  // CHECK:STDERR:   w.(c d e ());
  // CHECK:STDERR:        ^
  // CHECK:STDERR:
  w.(c d e ());

  // CHECK:STDERR: fail_tuple_instead_of_compound_member_access.carbon:[[@LINE+4]]:7: ERROR: Expected expression.
  // CHECK:STDERR:   v->();
  // CHECK:STDERR:       ^
  // CHECK:STDERR:
  v->();

  // CHECK:STDERR: fail_tuple_instead_of_compound_member_access.carbon:[[@LINE+4]]:8: ERROR: Expected `)`.
  // CHECK:STDERR:   u->(f,);
  // CHECK:STDERR:        ^
  // CHECK:STDERR:
  u->(f,);

  // CHECK:STDERR: fail_tuple_instead_of_compound_member_access.carbon:[[@LINE+4]]:8: ERROR: Expected `)`.
  // CHECK:STDERR:   t->(2, 3);
  // CHECK:STDERR:        ^
  // CHECK:STDERR:
  t->(2, 3);

  // CHECK:STDERR: fail_tuple_instead_of_compound_member_access.carbon:[[@LINE+3]]:9: ERROR: Expected `)`.
  // CHECK:STDERR:   s->(g h i ());
  // CHECK:STDERR:         ^
  s->(g h i ());
}

// CHECK:STDOUT: - filename: fail_tuple_instead_of_compound_member_access.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'F'},
// CHECK:STDOUT:           {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:         {kind: 'TuplePattern', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'x'},
// CHECK:STDOUT:             {kind: 'ParenExprStart', text: '('},
// CHECK:STDOUT:             {kind: 'InvalidParse', text: ')', has_error: yes},
// CHECK:STDOUT:           {kind: 'ParenExpr', text: ')', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:         {kind: 'MemberAccessExpr', text: '.', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 6},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'y'},
// CHECK:STDOUT:             {kind: 'ParenExprStart', text: '('},
// CHECK:STDOUT:             {kind: 'IntLiteral', text: '1'},
// CHECK:STDOUT:           {kind: 'ParenExpr', text: ')', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:         {kind: 'MemberAccessExpr', text: '.', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 6},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'z'},
// CHECK:STDOUT:             {kind: 'ParenExprStart', text: '('},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'a'},
// CHECK:STDOUT:           {kind: 'ParenExpr', text: ')', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:         {kind: 'MemberAccessExpr', text: '.', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 6},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'w'},
// CHECK:STDOUT:             {kind: 'ParenExprStart', text: '('},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'c'},
// CHECK:STDOUT:           {kind: 'ParenExpr', text: ')', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:         {kind: 'MemberAccessExpr', text: '.', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 6},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'v'},
// CHECK:STDOUT:             {kind: 'ParenExprStart', text: '('},
// CHECK:STDOUT:             {kind: 'InvalidParse', text: ')', has_error: yes},
// CHECK:STDOUT:           {kind: 'ParenExpr', text: ')', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:         {kind: 'PointerMemberAccessExpr', text: '->', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 6},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'u'},
// CHECK:STDOUT:             {kind: 'ParenExprStart', text: '('},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'f'},
// CHECK:STDOUT:           {kind: 'ParenExpr', text: ')', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:         {kind: 'PointerMemberAccessExpr', text: '->', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 6},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 't'},
// CHECK:STDOUT:             {kind: 'ParenExprStart', text: '('},
// CHECK:STDOUT:             {kind: 'IntLiteral', text: '2'},
// CHECK:STDOUT:           {kind: 'ParenExpr', text: ')', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:         {kind: 'PointerMemberAccessExpr', text: '->', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 6},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 's'},
// CHECK:STDOUT:             {kind: 'ParenExprStart', text: '('},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'g'},
// CHECK:STDOUT:           {kind: 'ParenExpr', text: ')', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:         {kind: 'PointerMemberAccessExpr', text: '->', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 6},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 54},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
