// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/operators/fail_precedence_as.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/operators/fail_precedence_as.carbon

fn F(n: i32) {
  // No ordering between `not` and `as`.
  // CHECK:STDERR: fail_precedence_as.carbon:[[@LINE+4]]:12: error: parentheses are required to disambiguate operator precedence [OperatorRequiresParentheses]
  // CHECK:STDERR:   not true as bool;
  // CHECK:STDERR:            ^~
  // CHECK:STDERR:
  not true as bool;

  // No ordering between mathematical binary operators and `as`.
  // CHECK:STDERR: fail_precedence_as.carbon:[[@LINE+4]]:9: error: parentheses are required to disambiguate operator precedence [OperatorRequiresParentheses]
  // CHECK:STDERR:   1 + 1 as i32;
  // CHECK:STDERR:         ^~
  // CHECK:STDERR:
  1 + 1 as i32;
  // CHECK:STDERR: fail_precedence_as.carbon:[[@LINE+4]]:9: error: parentheses are required to disambiguate operator precedence [OperatorRequiresParentheses]
  // CHECK:STDERR:   5 % 2 as i32;
  // CHECK:STDERR:         ^~
  // CHECK:STDERR:
  5 % 2 as i32;

  // No ordering between logical binary operators and `as`.
  // CHECK:STDERR: fail_precedence_as.carbon:[[@LINE+4]]:12: error: parentheses are required to disambiguate operator precedence [OperatorRequiresParentheses]
  // CHECK:STDERR:   3 as i32 and true;
  // CHECK:STDERR:            ^~~
  // CHECK:STDERR:
  3 as i32 and true;
  // CHECK:STDERR: fail_precedence_as.carbon:[[@LINE+4]]:14: error: parentheses are required to disambiguate operator precedence [OperatorRequiresParentheses]
  // CHECK:STDERR:   false or 4 as i32;
  // CHECK:STDERR:              ^~
  // CHECK:STDERR:
  false or 4 as i32;

  // No ordering between relational binary operators and `as`.
  // CHECK:STDERR: fail_precedence_as.carbon:[[@LINE+4]]:12: error: parentheses are required to disambiguate operator precedence [OperatorRequiresParentheses]
  // CHECK:STDERR:   6 as i32 == 7;
  // CHECK:STDERR:            ^~
  // CHECK:STDERR:
  6 as i32 == 7;
  // CHECK:STDERR: fail_precedence_as.carbon:[[@LINE+4]]:10: error: parentheses are required to disambiguate operator precedence [OperatorRequiresParentheses]
  // CHECK:STDERR:   8 <= 9 as i32;
  // CHECK:STDERR:          ^~
  // CHECK:STDERR:
  8 <= 9 as i32;
}

// CHECK:STDOUT: - filename: fail_precedence_as.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   │ │ ╭─IdentifierNameNotBeforeSignature 'n'
// CHECK:STDOUT:   │   │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │   │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │   │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │     ╭─BoolLiteralTrue 'true'
// CHECK:STDOUT:   │ │   ╭─PrefixOperatorNot 'not'
// CHECK:STDOUT:   │ │   ├─BoolTypeLiteral 'bool'
// CHECK:STDOUT:   │ │ ╭─InfixOperatorAs 'as' ❌
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │     ╭─IntLiteral '1'
// CHECK:STDOUT:   │ │     ├─IntLiteral '1'
// CHECK:STDOUT:   │ │   ╭─InfixOperatorPlus '+'
// CHECK:STDOUT:   │ │   ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ ╭─InfixOperatorAs 'as' ❌
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │     ╭─IntLiteral '5'
// CHECK:STDOUT:   │ │     ├─IntLiteral '2'
// CHECK:STDOUT:   │ │   ╭─InfixOperatorPercent '%'
// CHECK:STDOUT:   │ │   ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ ╭─InfixOperatorAs 'as' ❌
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │       ╭─IntLiteral '3'
// CHECK:STDOUT:   │ │       ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │     ╭─InfixOperatorAs 'as'
// CHECK:STDOUT:   │ │   ╭─ShortCircuitOperandAnd 'and' ❌
// CHECK:STDOUT:   │ │   ├─BoolLiteralTrue 'true'
// CHECK:STDOUT:   │ │ ╭─ShortCircuitOperatorAnd 'and' ❌
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │       ╭─BoolLiteralFalse 'false'
// CHECK:STDOUT:   │ │     ╭─ShortCircuitOperandOr 'or'
// CHECK:STDOUT:   │ │     ├─IntLiteral '4'
// CHECK:STDOUT:   │ │   ╭─ShortCircuitOperatorOr 'or'
// CHECK:STDOUT:   │ │   ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ ╭─InfixOperatorAs 'as' ❌
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │     ╭─IntLiteral '6'
// CHECK:STDOUT:   │ │     ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │   ╭─InfixOperatorAs 'as'
// CHECK:STDOUT:   │ │   ├─IntLiteral '7'
// CHECK:STDOUT:   │ │ ╭─InfixOperatorEqualEqual '==' ❌
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │     ╭─IntLiteral '8'
// CHECK:STDOUT:   │ │     ├─IntLiteral '9'
// CHECK:STDOUT:   │ │   ╭─InfixOperatorLessEqual '<='
// CHECK:STDOUT:   │ │   ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ ╭─InfixOperatorAs 'as' ❌
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
