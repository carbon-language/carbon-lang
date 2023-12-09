// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() {
  var n: i32 = 0;
  // TODO: It'd be nice to produce a custom diagnostic here.
  // CHECK:STDERR: fail_postincrement.carbon:[[@LINE+3]]:4: ERROR: Expected `;` after expression statement.
  // CHECK:STDERR:   n++;
  // CHECK:STDERR:    ^~
  n++;
  // CHECK:STDERR: fail_postincrement.carbon:[[@LINE+3]]:4: ERROR: Expected `;` after expression statement.
  // CHECK:STDERR:   n--;
  // CHECK:STDERR:    ^~
  n--;
}

// CHECK:STDOUT: - filename: fail_postincrement.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'F'},
// CHECK:STDOUT:           {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:         {kind: 'TuplePattern', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:           {kind: 'IdentifierName', text: 'n'},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'BindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'IntLiteral', text: '0'},
// CHECK:STDOUT:       {kind: 'VariableDecl', text: ';', subtree_size: 7},
// CHECK:STDOUT:         {kind: 'IdentifierNameExpr', text: 'n'},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:         {kind: 'IdentifierNameExpr', text: 'n'},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 17},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
