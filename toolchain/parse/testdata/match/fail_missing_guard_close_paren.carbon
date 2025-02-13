// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/match/fail_missing_guard_close_paren.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/match/fail_missing_guard_close_paren.carbon

fn f() -> i32 {
  match (true) {
    // CHECK:STDERR: fail_missing_guard_close_paren.carbon:[[@LINE+8]]:21: error: opening symbol without a corresponding closing symbol [UnmatchedOpening]
    // CHECK:STDERR:     case x: bool if (false => { return 1; }
    // CHECK:STDERR:                     ^
    // CHECK:STDERR:
    // CHECK:STDERR: fail_missing_guard_close_paren.carbon:[[@LINE+4]]:28: error: expected `)` [ExpectedMatchCaseGuardCloseParen]
    // CHECK:STDERR:     case x: bool if (false => { return 1; }
    // CHECK:STDERR:                            ^~
    // CHECK:STDERR:
    case x: bool if (false => { return 1; }
  }
  return 0;
}

// CHECK:STDOUT: - filename: fail_missing_guard_close_paren.carbon
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
// CHECK:STDOUT:               {kind: 'BoolLiteralFalse', text: 'false'},
// CHECK:STDOUT:             {kind: 'MatchCaseGuard', text: '=>', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:             {kind: 'MatchCaseEqualGreater', text: '}', has_error: yes},
// CHECK:STDOUT:           {kind: 'MatchCaseStart', text: '}', has_error: yes, subtree_size: 10},
// CHECK:STDOUT:         {kind: 'MatchCase', text: '}', has_error: yes, subtree_size: 11},
// CHECK:STDOUT:       {kind: 'MatchStatement', text: '}', subtree_size: 17},
// CHECK:STDOUT:         {kind: 'ReturnStatementStart', text: 'return'},
// CHECK:STDOUT:         {kind: 'IntLiteral', text: '0'},
// CHECK:STDOUT:       {kind: 'ReturnStatement', text: ';', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 28},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
