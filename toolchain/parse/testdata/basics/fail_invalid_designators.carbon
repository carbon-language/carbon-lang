// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// NOTE: Move to its own directory when more tests are added.
fn F() {
  // CHECK:STDERR: fail_invalid_designators.carbon:[[@LINE+3]]:5: ERROR: Expected identifier after `.`.
  // CHECK:STDERR:   a.;
  // CHECK:STDERR:     ^
  a.;
  // CHECK:STDERR: fail_invalid_designators.carbon:[[@LINE+3]]:5: ERROR: Expected identifier after `.`.
  // CHECK:STDERR:   a.fn;
  // CHECK:STDERR:     ^~
  a.fn;
  // CHECK:STDERR: fail_invalid_designators.carbon:[[@LINE+3]]:5: ERROR: Expected identifier after `.`.
  // CHECK:STDERR:   a.42;
  // CHECK:STDERR:     ^~
  a.42;
}

// CHECK:STDOUT: - filename: fail_invalid_designators.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'F'},
// CHECK:STDOUT:           {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:         {kind: 'TuplePattern', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'a'},
// CHECK:STDOUT:           {kind: 'IdentifierName', text: ';', has_error: yes},
// CHECK:STDOUT:         {kind: 'MemberAccessExpr', text: '.', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'a'},
// CHECK:STDOUT:           {kind: 'IdentifierName', text: 'fn', has_error: yes},
// CHECK:STDOUT:         {kind: 'MemberAccessExpr', text: '.', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'a'},
// CHECK:STDOUT:           {kind: 'IdentifierName', text: '42', has_error: yes},
// CHECK:STDOUT:         {kind: 'MemberAccessExpr', text: '.', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 18},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
