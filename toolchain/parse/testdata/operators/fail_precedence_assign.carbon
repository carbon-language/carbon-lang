// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/operators/fail_precedence_assign.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/operators/fail_precedence_assign.carbon

fn F() {
  var a: i32;
  // Assignment can only appear as an expression statement, not as a subexpression.
  // TODO: Produce the "can only be used as a complete statement" diagnostic in each of these cases.
  // CHECK:STDERR: fail_precedence_assign.carbon:[[@LINE+4]]:10: error: expected `,` or `)` [UnexpectedTokenAfterListElement]
  // CHECK:STDERR:   1 + (a = 1);
  // CHECK:STDERR:          ^
  // CHECK:STDERR:
  1 + (a = 1);
  // CHECK:STDERR: fail_precedence_assign.carbon:[[@LINE+8]]:19: error: expected `else` after `if ... then ...` [ExpectedElseAfterIf]
  // CHECK:STDERR:   (if true then a += 1 else a /= 2);
  // CHECK:STDERR:                   ^~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_precedence_assign.carbon:[[@LINE+4]]:19: error: expected `,` or `)` [UnexpectedTokenAfterListElement]
  // CHECK:STDERR:   (if true then a += 1 else a /= 2);
  // CHECK:STDERR:                   ^~
  // CHECK:STDERR:
  (if true then a += 1 else a /= 2);
  // CHECK:STDERR: fail_precedence_assign.carbon:[[@LINE+4]]:7: error: operator `++` can only be used as a complete statement [StatementOperatorAsSubExpr]
  // CHECK:STDERR:   a + ++a;
  // CHECK:STDERR:       ^~
  // CHECK:STDERR:
  a + ++a;
  // CHECK:STDERR: fail_precedence_assign.carbon:[[@LINE+4]]:5: error: parentheses are required around this unary `if` operator [UnaryOperatorRequiresParentheses]
  // CHECK:STDERR:   ++if c then a else b;
  // CHECK:STDERR:     ^~
  // CHECK:STDERR:
  ++if c then a else b;
}

// CHECK:STDOUT: - filename: fail_precedence_assign.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'F'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeParams', text: 'a'},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'VarBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'VariablePattern', text: 'var', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'VariableDecl', text: ';', subtree_size: 6},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '1'},
// CHECK:STDOUT:             {kind: 'ParenExprStart', text: '('},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'a'},
// CHECK:STDOUT:           {kind: 'ParenExpr', text: ')', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:         {kind: 'InfixOperatorPlus', text: '+', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 6},
// CHECK:STDOUT:           {kind: 'ParenExprStart', text: '('},
// CHECK:STDOUT:               {kind: 'BoolLiteralTrue', text: 'true'},
// CHECK:STDOUT:             {kind: 'IfExprIf', text: 'if', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'IdentifierNameExpr', text: 'a'},
// CHECK:STDOUT:             {kind: 'IfExprThen', text: 'then', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'InvalidParse', text: '+=', has_error: yes},
// CHECK:STDOUT:           {kind: 'IfExprElse', text: 'if', has_error: yes, subtree_size: 6},
// CHECK:STDOUT:         {kind: 'ParenExpr', text: ')', has_error: yes, subtree_size: 8},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 9},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'a'},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'a'},
// CHECK:STDOUT:           {kind: 'PrefixOperatorPlusPlus', text: '++', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'InfixOperatorPlus', text: '+', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 5},
// CHECK:STDOUT:               {kind: 'IdentifierNameExpr', text: 'c'},
// CHECK:STDOUT:             {kind: 'IfExprIf', text: 'if', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'IdentifierNameExpr', text: 'a'},
// CHECK:STDOUT:             {kind: 'IfExprThen', text: 'then', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'b'},
// CHECK:STDOUT:           {kind: 'IfExprElse', text: 'else', subtree_size: 6},
// CHECK:STDOUT:         {kind: 'PrefixOperatorPlusPlus', text: '++', subtree_size: 7},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 8},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 40},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
