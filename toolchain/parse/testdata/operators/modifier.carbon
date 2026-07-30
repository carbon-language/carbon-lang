// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/operators/modifier.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/operators/modifier.carbon

// --- valid.carbon

fn F() {
  a unsafe as b;
}

// --- fail_invalid.carbon

fn G() {
  // CHECK:STDERR: fail_invalid.carbon:[[@LINE+4]]:5: error: `unsafe` not allowed on operator `or` [ModifierNotAllowedOnOperator]
  // CHECK:STDERR:   a unsafe or b;
  // CHECK:STDERR:     ^~~~~~
  // CHECK:STDERR:
  a unsafe or b;

  // CHECK:STDERR: fail_invalid.carbon:[[@LINE+4]]:5: error: `unsafe` not allowed on operator `*` [ModifierNotAllowedOnOperator]
  // CHECK:STDERR:   a unsafe * b;
  // CHECK:STDERR:     ^~~~~~
  // CHECK:STDERR:
  a unsafe * b;

  // CHECK:STDERR: fail_invalid.carbon:[[@LINE+4]]:5: error: expected `;` after expression statement [ExpectedExprSemi]
  // CHECK:STDERR:   a unsafe;
  // CHECK:STDERR:     ^~~~~~
  // CHECK:STDERR:
  a unsafe;

  // CHECK:STDERR: fail_invalid.carbon:[[@LINE+4]]:3: error: expected expression [ExpectedExpr]
  // CHECK:STDERR:   unsafe;
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR:
  unsafe;

  // TODO: It'd be nice to underline both tokens in `unsafe as` here.
  // CHECK:STDERR: fail_invalid.carbon:[[@LINE+8]]:9: error: parentheses are required to disambiguate operator precedence [OperatorRequiresParentheses]
  // CHECK:STDERR:   a + b unsafe as c * d;
  // CHECK:STDERR:         ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_invalid.carbon:[[@LINE+4]]:21: error: parentheses are required to disambiguate operator precedence [OperatorRequiresParentheses]
  // CHECK:STDERR:   a + b unsafe as c * d;
  // CHECK:STDERR:                     ^
  // CHECK:STDERR:
  a + b unsafe as c * d;
}

// CHECK:STDOUT: - filename: valid.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │     ╭─IdentifierNameExpr 'a'
// CHECK:STDOUT:   │ │   ╭─UnsafeModifier 'unsafe'
// CHECK:STDOUT:   │ │   ├─IdentifierNameExpr 'b'
// CHECK:STDOUT:   │ │ ╭─InfixOperatorAs 'as'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_invalid.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'G'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │     ╭─IdentifierNameExpr 'a'
// CHECK:STDOUT:   │ │   ╭─ShortCircuitOperandOr 'or' has_error
// CHECK:STDOUT:   │ │   ├─IdentifierNameExpr 'b'
// CHECK:STDOUT:   │ │ ╭─ShortCircuitOperatorOr 'or' has_error
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameExpr 'a'
// CHECK:STDOUT:   │ │   ├─IdentifierNameExpr 'b'
// CHECK:STDOUT:   │ │ ╭─InfixOperatorStar '*' has_error
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   │ │ ╭─IdentifierNameExpr 'a'
// CHECK:STDOUT:   │ ├─ExprStatement ';' has_error
// CHECK:STDOUT:   │ │ ╭─InvalidParse 'unsafe' has_error
// CHECK:STDOUT:   │ ├─ExprStatement ';' has_error
// CHECK:STDOUT:   │ │         ╭─IdentifierNameExpr 'a'
// CHECK:STDOUT:   │ │         ├─IdentifierNameExpr 'b'
// CHECK:STDOUT:   │ │       ╭─InfixOperatorPlus '+'
// CHECK:STDOUT:   │ │     ╭─UnsafeModifier 'unsafe' has_error
// CHECK:STDOUT:   │ │     ├─IdentifierNameExpr 'c'
// CHECK:STDOUT:   │ │   ╭─InfixOperatorAs 'as' has_error
// CHECK:STDOUT:   │ │   ├─IdentifierNameExpr 'd'
// CHECK:STDOUT:   │ │ ╭─InfixOperatorStar '*' has_error
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
