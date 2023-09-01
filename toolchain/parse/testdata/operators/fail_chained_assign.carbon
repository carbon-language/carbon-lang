// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() {
  var a: i32;
  var b: i32;

  // TODO: Produce a custom diagnostic for this.
  // CHECK:STDERR: fail_chained_assign.carbon:[[@LINE+3]]:9: Operator `=` can only be used as a complete statement.
  // CHECK:STDERR:   a = b = 1;
  // CHECK:STDERR:         ^
  a = b = 1;
}

// CHECK:STDOUT: [
// CHECK:STDOUT:     {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:     {kind: 'Name', text: 'F'},
// CHECK:STDOUT:       {kind: 'ParameterListStart', text: '('},
// CHECK:STDOUT:     {kind: 'ParameterList', text: ')', subtree_size: 2},
// CHECK:STDOUT:   {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:       {kind: 'Name', text: 'a'},
// CHECK:STDOUT:       {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:     {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:   {kind: 'VariableDeclaration', text: ';', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:       {kind: 'Name', text: 'b'},
// CHECK:STDOUT:       {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:     {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:   {kind: 'VariableDeclaration', text: ';', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'NameExpression', text: 'a'},
// CHECK:STDOUT:         {kind: 'NameExpression', text: 'b'},
// CHECK:STDOUT:       {kind: 'InfixOperator', text: '=', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'Literal', text: '1'},
// CHECK:STDOUT:     {kind: 'InfixOperator', text: '=', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:   {kind: 'ExpressionStatement', text: ';', subtree_size: 6},
// CHECK:STDOUT: {kind: 'FunctionDefinition', text: '}', subtree_size: 22},
// CHECK:STDOUT: {kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]
