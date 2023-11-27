// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() {
  var a: i32;
  // Assignment can only appear as an expression statement, not as a subexpression.
  // TODO: Produce the "can only be used as a complete statement" diagnostic in each of these cases.
  // CHECK:STDERR: fail_precedence_assign.carbon:[[@LINE+3]]:10: ERROR: Expected `,` or `)`.
  // CHECK:STDERR:   1 + (a = 1);
  // CHECK:STDERR:          ^
  1 + (a = 1);
  // CHECK:STDERR: fail_precedence_assign.carbon:[[@LINE+6]]:19: ERROR: Expected `else` after `if ... then ...`.
  // CHECK:STDERR:   (if true then a += 1 else a /= 2);
  // CHECK:STDERR:                   ^~
  // CHECK:STDERR: fail_precedence_assign.carbon:[[@LINE+3]]:19: ERROR: Expected `,` or `)`.
  // CHECK:STDERR:   (if true then a += 1 else a /= 2);
  // CHECK:STDERR:                   ^~
  (if true then a += 1 else a /= 2);
  // CHECK:STDERR: fail_precedence_assign.carbon:[[@LINE+3]]:7: ERROR: Operator `++` can only be used as a complete statement.
  // CHECK:STDERR:   a + ++a;
  // CHECK:STDERR:       ^~
  a + ++a;
  // CHECK:STDERR: fail_precedence_assign.carbon:[[@LINE+3]]:5: ERROR: Parentheses are required around this unary `if` operator.
  // CHECK:STDERR:   ++if c then a else b;
  // CHECK:STDERR:     ^~
  ++if c then a else b;
}

// CHECK:STDOUT: - filename: fail_precedence_assign.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'Name', text: 'F'},
// CHECK:STDOUT:           {kind: 'ParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:           {kind: 'Name', text: 'a'},
// CHECK:STDOUT:           {kind: 'IntegerTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'VariableDecl', text: ';', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'IntegerLiteral', text: '1'},
// CHECK:STDOUT:             {kind: 'ParenExprOrTupleLiteralStart', text: '('},
// CHECK:STDOUT:             {kind: 'NameExpr', text: 'a'},
// CHECK:STDOUT:           {kind: 'ParenExpr', text: ')', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:         {kind: 'InfixOperator', text: '+', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 6},
// CHECK:STDOUT:           {kind: 'ParenExprOrTupleLiteralStart', text: '('},
// CHECK:STDOUT:               {kind: 'BoolLiteralTrue', text: 'true'},
// CHECK:STDOUT:             {kind: 'IfExprIf', text: 'if', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'NameExpr', text: 'a'},
// CHECK:STDOUT:             {kind: 'IfExprThen', text: 'then', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'InvalidParse', text: '+=', has_error: yes},
// CHECK:STDOUT:           {kind: 'IfExprElse', text: 'if', has_error: yes, subtree_size: 6},
// CHECK:STDOUT:         {kind: 'ParenExpr', text: ')', has_error: yes, subtree_size: 8},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 9},
// CHECK:STDOUT:           {kind: 'NameExpr', text: 'a'},
// CHECK:STDOUT:             {kind: 'NameExpr', text: 'a'},
// CHECK:STDOUT:           {kind: 'PrefixOperator', text: '++', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'InfixOperator', text: '+', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 5},
// CHECK:STDOUT:               {kind: 'NameExpr', text: 'c'},
// CHECK:STDOUT:             {kind: 'IfExprIf', text: 'if', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'NameExpr', text: 'a'},
// CHECK:STDOUT:             {kind: 'IfExprThen', text: 'then', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'NameExpr', text: 'b'},
// CHECK:STDOUT:           {kind: 'IfExprElse', text: 'else', subtree_size: 6},
// CHECK:STDOUT:         {kind: 'PrefixOperator', text: '++', subtree_size: 7},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 8},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 39},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
