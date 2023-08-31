// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() {
  var n: i32 = 0;
  // TODO: It'd be nice to produce a custom diagnostic here.
  // CHECK:STDERR: fail_postincrement.carbon:[[@LINE+3]]:4: Expected `;` after expression statement.
  // CHECK:STDERR:   n++;
  // CHECK:STDERR:    ^
  n++;
  // CHECK:STDERR: fail_postincrement.carbon:[[@LINE+3]]:4: Expected `;` after expression statement.
  // CHECK:STDERR:   n--;
  // CHECK:STDERR:    ^
  n--;
}

// CHECK:STDOUT: [
// CHECK:STDOUT:     {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:     {kind: 'Name', text: 'F'},
// CHECK:STDOUT:       {kind: 'ParameterListStart', text: '('},
// CHECK:STDOUT:     {kind: 'ParameterList', text: ')', subtree_size: 2},
// CHECK:STDOUT:   {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:       {kind: 'Name', text: 'n'},
// CHECK:STDOUT:       {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:     {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:     {kind: 'Literal', text: '0'},
// CHECK:STDOUT:   {kind: 'VariableDeclaration', text: ';', subtree_size: 7},
// CHECK:STDOUT:     {kind: 'NameExpression', text: 'n'},
// CHECK:STDOUT:   {kind: 'ExpressionStatement', text: ';', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:     {kind: 'NameExpression', text: 'n'},
// CHECK:STDOUT:   {kind: 'ExpressionStatement', text: ';', has_error: yes, subtree_size: 2},
// CHECK:STDOUT: {kind: 'FunctionDefinition', text: '}', subtree_size: 17},
// CHECK:STDOUT: {kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]
