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
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'F'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'a'},
// CHECK:STDOUT:           {kind: 'UnsafeModifier', text: 'unsafe', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'b'},
// CHECK:STDOUT:         {kind: 'InfixOperatorAs', text: 'as', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 11},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
// CHECK:STDOUT: - filename: fail_invalid.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'G'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'a'},
// CHECK:STDOUT:           {kind: 'ShortCircuitOperandOr', text: 'or', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'b'},
// CHECK:STDOUT:         {kind: 'ShortCircuitOperatorOr', text: 'or', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'a'},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'b'},
// CHECK:STDOUT:         {kind: 'InfixOperatorStar', text: '*', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'IdentifierNameExpr', text: 'a'},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:         {kind: 'InvalidParse', text: 'unsafe', has_error: yes},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:                 {kind: 'IdentifierNameExpr', text: 'a'},
// CHECK:STDOUT:                 {kind: 'IdentifierNameExpr', text: 'b'},
// CHECK:STDOUT:               {kind: 'InfixOperatorPlus', text: '+', subtree_size: 3},
// CHECK:STDOUT:             {kind: 'UnsafeModifier', text: 'unsafe', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'c'},
// CHECK:STDOUT:           {kind: 'InfixOperatorAs', text: 'as', has_error: yes, subtree_size: 6},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'd'},
// CHECK:STDOUT:         {kind: 'InfixOperatorStar', text: '*', has_error: yes, subtree_size: 8},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 9},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 28},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
