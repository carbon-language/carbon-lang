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
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'AndWhere'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:                 {kind: 'BoolLiteralTrue', text: 'true'},
// CHECK:STDOUT:               {kind: 'ShortCircuitOperandAnd', text: 'and', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'IdentifierNameExpr', text: 'T'},
// CHECK:STDOUT:             {kind: 'ShortCircuitOperatorAnd', text: 'and', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'WhereOperand', text: 'where', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeParams', text: 'A'},
// CHECK:STDOUT:             {kind: 'DesignatorExpr', text: '.', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'B'},
// CHECK:STDOUT:           {kind: 'RequirementEqual', text: '=', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'WhereExpr', text: 'where', has_error: yes, subtree_size: 10},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', has_error: yes, subtree_size: 11},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 17},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_where_or.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'WhereOr'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'U'},
// CHECK:STDOUT:           {kind: 'WhereOperand', text: 'where', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeParams', text: 'C'},
// CHECK:STDOUT:             {kind: 'DesignatorExpr', text: '.', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'D'},
// CHECK:STDOUT:           {kind: 'RequirementImpls', text: 'impls', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'WhereExpr', text: 'where', subtree_size: 7},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', has_error: yes, subtree_size: 8},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 14},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_equals_where.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'EqualsWhere'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:               {kind: 'IdentifierNameExpr', text: 'e'},
// CHECK:STDOUT:               {kind: 'IdentifierNameExpr', text: 'V'},
// CHECK:STDOUT:             {kind: 'InfixOperatorEqualEqual', text: '==', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'WhereOperand', text: 'where', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeParams', text: 'G'},
// CHECK:STDOUT:             {kind: 'DesignatorExpr', text: '.', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'FloatTypeLiteral', text: 'f32'},
// CHECK:STDOUT:           {kind: 'RequirementEqualEqual', text: '==', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'WhereExpr', text: 'where', has_error: yes, subtree_size: 9},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', has_error: yes, subtree_size: 10},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 16},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_where_neq.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'WhereNEq'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'W'},
// CHECK:STDOUT:           {kind: 'WhereOperand', text: 'where', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeParams', text: 'H'},
// CHECK:STDOUT:             {kind: 'DesignatorExpr', text: '.', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'BoolTypeLiteral', text: 'bool'},
// CHECK:STDOUT:           {kind: 'RequirementEqual', text: '=', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'WhereExpr', text: 'where', subtree_size: 7},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', has_error: yes, subtree_size: 8},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 14},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_less_requirement.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'LessRequirement'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'X'},
// CHECK:STDOUT:           {kind: 'WhereOperand', text: 'where', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeParams', text: 'J'},
// CHECK:STDOUT:           {kind: 'DesignatorExpr', text: '.', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'WhereExpr', text: 'where', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', has_error: yes, subtree_size: 6},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 12},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_as_where.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'AsWhere'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:               {kind: 'IdentifierNameExpr', text: 'Y'},
// CHECK:STDOUT:               {kind: 'IdentifierNameExpr', text: 'Z'},
// CHECK:STDOUT:             {kind: 'InfixOperatorAs', text: 'as', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'WhereOperand', text: 'where', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeParams', text: 'L'},
// CHECK:STDOUT:             {kind: 'DesignatorExpr', text: '.', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'M'},
// CHECK:STDOUT:           {kind: 'RequirementImpls', text: 'impls', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'WhereExpr', text: 'where', has_error: yes, subtree_size: 9},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', has_error: yes, subtree_size: 10},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 16},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_where_as.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'WhereAs'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'S'},
// CHECK:STDOUT:           {kind: 'WhereOperand', text: 'where', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeParams', text: 'N'},
// CHECK:STDOUT:             {kind: 'DesignatorExpr', text: '.', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'RequirementEqualEqual', text: '==', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'WhereExpr', text: 'where', subtree_size: 7},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', has_error: yes, subtree_size: 8},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 14},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_as_in_argument.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'AsInArgument'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'R'},
// CHECK:STDOUT:           {kind: 'WhereOperand', text: 'where', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeParams', text: 'O'},
// CHECK:STDOUT:           {kind: 'DesignatorExpr', text: '.', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'WhereExpr', text: 'where', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', has_error: yes, subtree_size: 6},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 12},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_greater_in_argument.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'GreaterInArgument'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'Q'},
// CHECK:STDOUT:           {kind: 'WhereOperand', text: 'where', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeParams', text: 'AA'},
// CHECK:STDOUT:           {kind: 'DesignatorExpr', text: '.', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'WhereExpr', text: 'where', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', has_error: yes, subtree_size: 6},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 12},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
