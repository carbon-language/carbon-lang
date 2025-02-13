// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/match/fail_missing_guard_parens_only_parse_errors.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/match/fail_missing_guard_parens_only_parse_errors.carbon

fn f() -> i32 {
  match (true) {
    // CHECK:STDERR: fail_missing_guard_parens_only_parse_errors.carbon:[[@LINE+4]]:33: error: expected `)` [ExpectedMatchCaseGuardCloseParen]
    // CHECK:STDERR:     case x: bool if (x == false => { return 1; }
    // CHECK:STDERR:                                 ^~
    // CHECK:STDERR:
    case x: bool if (x == false => { return 1; }
    case y: bool if (not y) => { return 2; }
    // CHECK:STDERR: fail_missing_guard_parens_only_parse_errors.carbon:[[@LINE+4]]:21: error: expected `(` after `if` [ExpectedMatchCaseGuardOpenParen]
    // CHECK:STDERR:     case z: bool if true or false) => { return 3; }
    // CHECK:STDERR:                     ^~~~
    // CHECK:STDERR:
    case z: bool if true or false) => { return 3; }
  }
  return 0;
}

// CHECK:STDOUT: - filename: fail_missing_guard_parens_only_parse_errors.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'f'},
// CHECK:STDOUT:           {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:         {kind: 'TuplePattern', text: ')', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'ReturnType', text: '->', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 7},
// CHECK:STDOUT:           {kind: 'MatchIntroducer', text: 'match'},
// CHECK:STDOUT:             {kind: 'MatchConditionStart', text: '('},
// CHECK:STDOUT:             {kind: 'BoolLiteralTrue', text: 'true'},
// CHECK:STDOUT:           {kind: 'MatchCondition', text: ')', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'MatchStatementStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:             {kind: 'MatchCaseIntroducer', text: 'case'},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeParams', text: 'x'},
// CHECK:STDOUT:               {kind: 'BoolTypeLiteral', text: 'bool'},
// CHECK:STDOUT:             {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:               {kind: 'MatchCaseGuardIntroducer', text: 'if'},
// CHECK:STDOUT:               {kind: 'MatchCaseGuardStart', text: '('},
// CHECK:STDOUT:                 {kind: 'IdentifierNameExpr', text: 'x'},
// CHECK:STDOUT:                 {kind: 'BoolLiteralFalse', text: 'false'},
// CHECK:STDOUT:               {kind: 'InfixOperatorEqualEqual', text: '==', subtree_size: 3},
// CHECK:STDOUT:             {kind: 'MatchCaseGuard', text: '=>', has_error: yes, subtree_size: 6},
// CHECK:STDOUT:             {kind: 'MatchCaseEqualGreater', text: 'case', has_error: yes},
// CHECK:STDOUT:           {kind: 'MatchCaseStart', text: 'case', has_error: yes, subtree_size: 12},
// CHECK:STDOUT:         {kind: 'MatchCase', text: 'case', has_error: yes, subtree_size: 13},
// CHECK:STDOUT:             {kind: 'MatchCaseIntroducer', text: 'case'},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeParams', text: 'z'},
// CHECK:STDOUT:               {kind: 'BoolTypeLiteral', text: 'bool'},
// CHECK:STDOUT:             {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:               {kind: 'MatchCaseGuardIntroducer', text: 'if'},
// CHECK:STDOUT:               {kind: 'MatchCaseGuardStart', text: 'true', has_error: yes},
// CHECK:STDOUT:               {kind: 'InvalidParse', text: 'true', has_error: yes},
// CHECK:STDOUT:             {kind: 'MatchCaseGuard', text: 'true', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:           {kind: 'MatchCaseStart', text: 'true', has_error: yes, subtree_size: 9},
// CHECK:STDOUT:         {kind: 'MatchCase', text: 'true', has_error: yes, subtree_size: 10},
// CHECK:STDOUT:       {kind: 'MatchStatement', text: '}', subtree_size: 29},
// CHECK:STDOUT:         {kind: 'ReturnStatementStart', text: 'return'},
// CHECK:STDOUT:         {kind: 'IntLiteral', text: '0'},
// CHECK:STDOUT:       {kind: 'ReturnStatement', text: ';', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 40},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
