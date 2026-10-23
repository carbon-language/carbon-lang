// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/operators/fail_variety.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/operators/fail_variety.carbon

fn F() {
  // CHECK:STDERR: fail_variety.carbon:[[@LINE+10]]:21: error: operator `=` can only be used as a complete statement [StatementOperatorAsSubExpr]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_variety.carbon:[[@LINE+8]]:29-30: error: parentheses are required to disambiguate operator precedence [OperatorRequiresParentheses]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_variety.carbon:[[@LINE+6]]:34: error: parentheses are required to disambiguate operator precedence [OperatorRequiresParentheses]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_variety.carbon:[[@LINE+4]]:38: error: parentheses are required to disambiguate operator precedence [OperatorRequiresParentheses]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_variety.carbon:[[@LINE+2]]:40-42: error: parentheses are required around this unary `not` operator [UnaryOperatorRequiresParentheses]
  // CHECK:STDERR:
  n = a * b + c * d = d * d << e & f - not g;
}

// CHECK:STDOUT: - filename: fail_variety.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │     ╭─IdentifierNameExpr 'n'
// CHECK:STDOUT:   │ │     │   ╭─IdentifierNameExpr 'a'
// CHECK:STDOUT:   │ │     │   ├─IdentifierNameExpr 'b'
// CHECK:STDOUT:   │ │     │ ╭─InfixOperatorStar '*'
// CHECK:STDOUT:   │ │     │ │ ╭─IdentifierNameExpr 'c'
// CHECK:STDOUT:   │ │     │ │ ├─IdentifierNameExpr 'd'
// CHECK:STDOUT:   │ │     │ ├─InfixOperatorStar '*'
// CHECK:STDOUT:   │ │     ├─InfixOperatorPlus '+'
// CHECK:STDOUT:   │ │   ╭─InfixOperatorEqual '='
// CHECK:STDOUT:   │ │   │       ╭─IdentifierNameExpr 'd'
// CHECK:STDOUT:   │ │   │       ├─IdentifierNameExpr 'd'
// CHECK:STDOUT:   │ │   │     ╭─InfixOperatorStar '*'
// CHECK:STDOUT:   │ │   │     ├─IdentifierNameExpr 'e'
// CHECK:STDOUT:   │ │   │   ╭─InfixOperatorLessLess '<<' has_error
// CHECK:STDOUT:   │ │   │   ├─IdentifierNameExpr 'f'
// CHECK:STDOUT:   │ │   │ ╭─InfixOperatorAmp '&' has_error
// CHECK:STDOUT:   │ │   │ │ ╭─IdentifierNameExpr 'g'
// CHECK:STDOUT:   │ │   │ ├─PrefixOperatorNot 'not'
// CHECK:STDOUT:   │ │   ├─InfixOperatorMinus '-' has_error
// CHECK:STDOUT:   │ │ ╭─InfixOperatorEqual '=' has_error
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
