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
  // CHECK:STDERR: fail_precedence_assign.carbon:[[@LINE+2]]:10: error: expected `,` or `)` [UnexpectedTokenAfterListElement]
  // CHECK:STDERR:
  1 + (a = 1);
  // CHECK:STDERR: fail_precedence_assign.carbon:[[@LINE+4]]:19-20: error: expected `else` after `if ... then ...` [ExpectedElseAfterIf]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_precedence_assign.carbon:[[@LINE+2]]:19-20: error: expected `,` or `)` [UnexpectedTokenAfterListElement]
  // CHECK:STDERR:
  (if true then a += 1 else a /= 2);
  // CHECK:STDERR: fail_precedence_assign.carbon:[[@LINE+2]]:7-8: error: operator `++` can only be used as a complete statement [StatementOperatorAsSubExpr]
  // CHECK:STDERR:
  a + ++a;
  // CHECK:STDERR: fail_precedence_assign.carbon:[[@LINE+2]]:5-6: error: parentheses are required around this unary `if` operator [UnaryOperatorRequiresParentheses]
  // CHECK:STDERR:
  ++if c then a else b;
}

// CHECK:STDOUT: - filename: fail_precedence_assign.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'a'
// CHECK:STDOUT:   │ │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │   ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ ├─VariableDecl ';'
// CHECK:STDOUT:   │ │   ╭─IntLiteral '1'
// CHECK:STDOUT:   │ │   │ ╭─ParenExprStart '('
// CHECK:STDOUT:   │ │   │ ├─IdentifierNameExpr 'a'
// CHECK:STDOUT:   │ │   ├─ParenExpr ')' has_error
// CHECK:STDOUT:   │ │ ╭─InfixOperatorPlus '+'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │   ╭─ParenExprStart '('
// CHECK:STDOUT:   │ │   │   ╭─BoolLiteralTrue 'true'
// CHECK:STDOUT:   │ │   │ ╭─IfExprIf 'if'
// CHECK:STDOUT:   │ │   │ │ ╭─IdentifierNameExpr 'a'
// CHECK:STDOUT:   │ │   │ ├─IfExprThen 'then'
// CHECK:STDOUT:   │ │   │ ├─InvalidParse '+=' has_error
// CHECK:STDOUT:   │ │   ├─IfExprElse 'if' has_error
// CHECK:STDOUT:   │ │ ╭─ParenExpr ')' has_error
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameExpr 'a'
// CHECK:STDOUT:   │ │   │ ╭─IdentifierNameExpr 'a'
// CHECK:STDOUT:   │ │   ├─PrefixOperatorPlusPlus '++'
// CHECK:STDOUT:   │ │ ╭─InfixOperatorPlus '+'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │       ╭─IdentifierNameExpr 'c'
// CHECK:STDOUT:   │ │     ╭─IfExprIf 'if'
// CHECK:STDOUT:   │ │     │ ╭─IdentifierNameExpr 'a'
// CHECK:STDOUT:   │ │     ├─IfExprThen 'then'
// CHECK:STDOUT:   │ │     ├─IdentifierNameExpr 'b'
// CHECK:STDOUT:   │ │   ╭─IfExprElse 'else'
// CHECK:STDOUT:   │ │ ╭─PrefixOperatorPlusPlus '++'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
