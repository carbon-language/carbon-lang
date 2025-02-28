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
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'F'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeParams', text: 'n'},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 8},
// CHECK:STDOUT:             {kind: 'BoolLiteralTrue', text: 'true'},
// CHECK:STDOUT:           {kind: 'PrefixOperatorNot', text: 'not', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'BoolTypeLiteral', text: 'bool'},
// CHECK:STDOUT:         {kind: 'InfixOperatorAs', text: 'as', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 5},
// CHECK:STDOUT:             {kind: 'IntLiteral', text: '1'},
// CHECK:STDOUT:             {kind: 'IntLiteral', text: '1'},
// CHECK:STDOUT:           {kind: 'InfixOperatorPlus', text: '+', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'InfixOperatorAs', text: 'as', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 6},
// CHECK:STDOUT:             {kind: 'IntLiteral', text: '5'},
// CHECK:STDOUT:             {kind: 'IntLiteral', text: '2'},
// CHECK:STDOUT:           {kind: 'InfixOperatorPercent', text: '%', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'InfixOperatorAs', text: 'as', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 6},
// CHECK:STDOUT:               {kind: 'IntLiteral', text: '3'},
// CHECK:STDOUT:               {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:             {kind: 'InfixOperatorAs', text: 'as', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'ShortCircuitOperandAnd', text: 'and', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:           {kind: 'BoolLiteralTrue', text: 'true'},
// CHECK:STDOUT:         {kind: 'ShortCircuitOperatorAnd', text: 'and', has_error: yes, subtree_size: 6},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 7},
// CHECK:STDOUT:               {kind: 'BoolLiteralFalse', text: 'false'},
// CHECK:STDOUT:             {kind: 'ShortCircuitOperandOr', text: 'or', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'IntLiteral', text: '4'},
// CHECK:STDOUT:           {kind: 'ShortCircuitOperatorOr', text: 'or', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'InfixOperatorAs', text: 'as', has_error: yes, subtree_size: 6},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 7},
// CHECK:STDOUT:             {kind: 'IntLiteral', text: '6'},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'InfixOperatorAs', text: 'as', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'IntLiteral', text: '7'},
// CHECK:STDOUT:         {kind: 'InfixOperatorEqualEqual', text: '==', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 6},
// CHECK:STDOUT:             {kind: 'IntLiteral', text: '8'},
// CHECK:STDOUT:             {kind: 'IntLiteral', text: '9'},
// CHECK:STDOUT:           {kind: 'InfixOperatorLessEqual', text: '<=', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'InfixOperatorAs', text: 'as', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 6},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 52},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
