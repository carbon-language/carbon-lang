// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/operators/fail_precedence_where.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/operators/fail_precedence_where.carbon

// --- fail_and_where.carbon

fn AndWhere() {
  // CHECK:STDERR: fail_and_where.carbon:[[@LINE+4]]:14: error: parentheses are required to disambiguate operator precedence [OperatorRequiresParentheses]
  // CHECK:STDERR:   true and T where .A = B;
  // CHECK:STDERR:              ^~~~~
  // CHECK:STDERR:
  true and T where .A = B;
}

// --- fail_where_or.carbon

fn WhereOr() {
  // CHECK:STDERR: fail_where_or.carbon:[[@LINE+4]]:22: error: expected `;` after expression statement [ExpectedExprSemi]
  // CHECK:STDERR:   U where .C impls D or false;
  // CHECK:STDERR:                      ^~
  // CHECK:STDERR:
  U where .C impls D or false;
}

// --- fail_equals_where.carbon

fn EqualsWhere() {
  // CHECK:STDERR: fail_equals_where.carbon:[[@LINE+4]]:10: error: parentheses are required to disambiguate operator precedence [OperatorRequiresParentheses]
  // CHECK:STDERR:   e == V where .G == f32;
  // CHECK:STDERR:          ^~~~~
  // CHECK:STDERR:
  e == V where .G == f32;
}

// --- fail_where_neq.carbon

fn WhereNEq() {
  // CHECK:STDERR: fail_where_neq.carbon:[[@LINE+4]]:21: error: expected `;` after expression statement [ExpectedExprSemi]
  // CHECK:STDERR:   W where .H = bool != i;
  // CHECK:STDERR:                     ^~
  // CHECK:STDERR:
  W where .H = bool != i;
}

// --- fail_less_requirement.carbon

fn LessRequirement() {
  // CHECK:STDERR: fail_less_requirement.carbon:[[@LINE+4]]:14: error: requirement should use `impls`, `=`, or `==` operator [ExpectedRequirementOperator]
  // CHECK:STDERR:   X where .J < .K;
  // CHECK:STDERR:              ^
  // CHECK:STDERR:
  X where .J < .K;
}

// --- fail_as_where.carbon

fn AsWhere() {
  // CHECK:STDERR: fail_as_where.carbon:[[@LINE+4]]:10: error: parentheses are required to disambiguate operator precedence [OperatorRequiresParentheses]
  // CHECK:STDERR:   Y as Z where .L impls M;
  // CHECK:STDERR:          ^~~~~
  // CHECK:STDERR:
  Y as Z where .L impls M;
}

// --- fail_where_as.carbon

fn WhereAs() {
  // CHECK:STDERR: fail_where_as.carbon:[[@LINE+4]]:21: error: expected `;` after expression statement [ExpectedExprSemi]
  // CHECK:STDERR:   S where .N == i32 as type;
  // CHECK:STDERR:                     ^~
  // CHECK:STDERR:
  S where .N == i32 as type;
}

// --- fail_as_in_argument.carbon

fn AsInArgument() {
  // CHECK:STDERR: fail_as_in_argument.carbon:[[@LINE+4]]:14: error: requirement should use `impls`, `=`, or `==` operator [ExpectedRequirementOperator]
  // CHECK:STDERR:   R where .O as type impls P;
  // CHECK:STDERR:              ^~
  // CHECK:STDERR:
  R where .O as type impls P;
}

// --- fail_greater_in_argument.carbon

fn GreaterInArgument() {
  // CHECK:STDERR: fail_greater_in_argument.carbon:[[@LINE+4]]:15: error: requirement should use `impls`, `=`, or `==` operator [ExpectedRequirementOperator]
  // CHECK:STDERR:   Q where .AA > .BB impls CC;
  // CHECK:STDERR:               ^
  // CHECK:STDERR:
  Q where .AA > .BB impls CC;
}

// CHECK:STDOUT: - filename: fail_and_where.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'AndWhere'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │         ╭─BoolLiteralTrue 'true'
// CHECK:STDOUT:   │ │       ╭─ShortCircuitOperandAnd 'and'
// CHECK:STDOUT:   │ │       ├─IdentifierNameExpr 'T'
// CHECK:STDOUT:   │ │     ╭─ShortCircuitOperatorAnd 'and'
// CHECK:STDOUT:   │ │   ╭─WhereOperand 'where' ❌
// CHECK:STDOUT:   │ │   │   ╭─IdentifierNameNotBeforeSignature 'A'
// CHECK:STDOUT:   │ │   │ ╭─DesignatorExpr '.'
// CHECK:STDOUT:   │ │   │ ├─IdentifierNameExpr 'B'
// CHECK:STDOUT:   │ │   ├─RequirementEqual '='
// CHECK:STDOUT:   │ │ ╭─WhereExpr 'where' ❌
// CHECK:STDOUT:   │ ├─ExprStatement ';' ❌
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_where_or.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'WhereOr'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │     ╭─IdentifierNameExpr 'U'
// CHECK:STDOUT:   │ │   ╭─WhereOperand 'where'
// CHECK:STDOUT:   │ │   │   ╭─IdentifierNameNotBeforeSignature 'C'
// CHECK:STDOUT:   │ │   │ ╭─DesignatorExpr '.'
// CHECK:STDOUT:   │ │   │ ├─IdentifierNameExpr 'D'
// CHECK:STDOUT:   │ │   ├─RequirementImpls 'impls'
// CHECK:STDOUT:   │ │ ╭─WhereExpr 'where'
// CHECK:STDOUT:   │ ├─ExprStatement ';' ❌
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_equals_where.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'EqualsWhere'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │       ╭─IdentifierNameExpr 'e'
// CHECK:STDOUT:   │ │       ├─IdentifierNameExpr 'V'
// CHECK:STDOUT:   │ │     ╭─InfixOperatorEqualEqual '=='
// CHECK:STDOUT:   │ │   ╭─WhereOperand 'where' ❌
// CHECK:STDOUT:   │ │   │   ╭─IdentifierNameNotBeforeSignature 'G'
// CHECK:STDOUT:   │ │   │ ╭─DesignatorExpr '.'
// CHECK:STDOUT:   │ │   │ ├─FloatTypeLiteral 'f32'
// CHECK:STDOUT:   │ │   ├─RequirementEqualEqual '=='
// CHECK:STDOUT:   │ │ ╭─WhereExpr 'where' ❌
// CHECK:STDOUT:   │ ├─ExprStatement ';' ❌
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_where_neq.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'WhereNEq'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │     ╭─IdentifierNameExpr 'W'
// CHECK:STDOUT:   │ │   ╭─WhereOperand 'where'
// CHECK:STDOUT:   │ │   │   ╭─IdentifierNameNotBeforeSignature 'H'
// CHECK:STDOUT:   │ │   │ ╭─DesignatorExpr '.'
// CHECK:STDOUT:   │ │   │ ├─BoolTypeLiteral 'bool'
// CHECK:STDOUT:   │ │   ├─RequirementEqual '='
// CHECK:STDOUT:   │ │ ╭─WhereExpr 'where'
// CHECK:STDOUT:   │ ├─ExprStatement ';' ❌
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_less_requirement.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'LessRequirement'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │     ╭─IdentifierNameExpr 'X'
// CHECK:STDOUT:   │ │   ╭─WhereOperand 'where'
// CHECK:STDOUT:   │ │   │ ╭─IdentifierNameNotBeforeSignature 'J'
// CHECK:STDOUT:   │ │   ├─DesignatorExpr '.'
// CHECK:STDOUT:   │ │ ╭─WhereExpr 'where' ❌
// CHECK:STDOUT:   │ ├─ExprStatement ';' ❌
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_as_where.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'AsWhere'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │       ╭─IdentifierNameExpr 'Y'
// CHECK:STDOUT:   │ │       ├─IdentifierNameExpr 'Z'
// CHECK:STDOUT:   │ │     ╭─InfixOperatorAs 'as'
// CHECK:STDOUT:   │ │   ╭─WhereOperand 'where' ❌
// CHECK:STDOUT:   │ │   │   ╭─IdentifierNameNotBeforeSignature 'L'
// CHECK:STDOUT:   │ │   │ ╭─DesignatorExpr '.'
// CHECK:STDOUT:   │ │   │ ├─IdentifierNameExpr 'M'
// CHECK:STDOUT:   │ │   ├─RequirementImpls 'impls'
// CHECK:STDOUT:   │ │ ╭─WhereExpr 'where' ❌
// CHECK:STDOUT:   │ ├─ExprStatement ';' ❌
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_where_as.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'WhereAs'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │     ╭─IdentifierNameExpr 'S'
// CHECK:STDOUT:   │ │   ╭─WhereOperand 'where'
// CHECK:STDOUT:   │ │   │   ╭─IdentifierNameNotBeforeSignature 'N'
// CHECK:STDOUT:   │ │   │ ╭─DesignatorExpr '.'
// CHECK:STDOUT:   │ │   │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │   ├─RequirementEqualEqual '=='
// CHECK:STDOUT:   │ │ ╭─WhereExpr 'where'
// CHECK:STDOUT:   │ ├─ExprStatement ';' ❌
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_as_in_argument.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'AsInArgument'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │     ╭─IdentifierNameExpr 'R'
// CHECK:STDOUT:   │ │   ╭─WhereOperand 'where'
// CHECK:STDOUT:   │ │   │ ╭─IdentifierNameNotBeforeSignature 'O'
// CHECK:STDOUT:   │ │   ├─DesignatorExpr '.'
// CHECK:STDOUT:   │ │ ╭─WhereExpr 'where' ❌
// CHECK:STDOUT:   │ ├─ExprStatement ';' ❌
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_greater_in_argument.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'GreaterInArgument'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │     ╭─IdentifierNameExpr 'Q'
// CHECK:STDOUT:   │ │   ╭─WhereOperand 'where'
// CHECK:STDOUT:   │ │   │ ╭─IdentifierNameNotBeforeSignature 'AA'
// CHECK:STDOUT:   │ │   ├─DesignatorExpr '.'
// CHECK:STDOUT:   │ │ ╭─WhereExpr 'where' ❌
// CHECK:STDOUT:   │ ├─ExprStatement ';' ❌
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
