// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() {
  var a: i32;
  // Assignment can only appear as an expression statement, not as a subexpression.
  // TODO: Produce the "can only be used as a complete statement" diagnostic in each of these cases.
  // CHECK:STDERR: fail_precedence_assign.carbon:[[@LINE+3]]:10: Expected `,` or `)`.
  // CHECK:STDERR:   1 + (a = 1);
  // CHECK:STDERR:          ^
  1 + (a = 1);
  // CHECK:STDERR: fail_precedence_assign.carbon:[[@LINE+6]]:19: Expected `else` after `if ... then ...`.
  // CHECK:STDERR:   (if true then a += 1 else a /= 2);
  // CHECK:STDERR:                   ^
  // CHECK:STDERR: fail_precedence_assign.carbon:[[@LINE+3]]:19: Expected `,` or `)`.
  // CHECK:STDERR:   (if true then a += 1 else a /= 2);
  // CHECK:STDERR:                   ^
  (if true then a += 1 else a /= 2);
  // CHECK:STDERR: fail_precedence_assign.carbon:[[@LINE+3]]:7: Operator `++` can only be used as a complete statement.
  // CHECK:STDERR:   a + ++a;
  // CHECK:STDERR:       ^
  a + ++a;
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
// CHECK:STDOUT:       {kind: 'Literal', text: '1'},
// CHECK:STDOUT:         {kind: 'ParenExpressionOrTupleLiteralStart', text: '('},
// CHECK:STDOUT:         {kind: 'NameExpression', text: 'a'},
// CHECK:STDOUT:       {kind: 'ParenExpression', text: ')', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'InfixOperator', text: '+', subtree_size: 5},
// CHECK:STDOUT:   {kind: 'ExpressionStatement', text: ';', subtree_size: 6},
// CHECK:STDOUT:       {kind: 'ParenExpressionOrTupleLiteralStart', text: '('},
// CHECK:STDOUT:           {kind: 'Literal', text: 'true'},
// CHECK:STDOUT:         {kind: 'IfExpressionIf', text: 'if', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'NameExpression', text: 'a'},
// CHECK:STDOUT:         {kind: 'IfExpressionThen', text: 'then', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'InvalidParse', text: '+=', has_error: yes},
// CHECK:STDOUT:       {kind: 'IfExpressionElse', text: 'if', has_error: yes, subtree_size: 6},
// CHECK:STDOUT:     {kind: 'ParenExpression', text: ')', has_error: yes, subtree_size: 8},
// CHECK:STDOUT:   {kind: 'ExpressionStatement', text: ';', subtree_size: 9},
// CHECK:STDOUT:       {kind: 'NameExpression', text: 'a'},
// CHECK:STDOUT:         {kind: 'NameExpression', text: 'a'},
// CHECK:STDOUT:       {kind: 'PrefixOperator', text: '++', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'InfixOperator', text: '+', subtree_size: 4},
// CHECK:STDOUT:   {kind: 'ExpressionStatement', text: ';', subtree_size: 5},
// CHECK:STDOUT: {kind: 'FunctionDefinition', text: '}', subtree_size: 31},
// CHECK:STDOUT: {kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]
