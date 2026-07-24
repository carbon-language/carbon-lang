// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/match/fail_missing_guard_open_paren.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/match/fail_missing_guard_open_paren.carbon

fn f() -> i32 {
  match (true) {
    // CHECK:STDERR: fail_missing_guard_open_paren.carbon:[[@LINE+7]]:26: error: closing symbol without a corresponding opening symbol [UnmatchedClosing]
    // CHECK:STDERR:     case x: bool if false) => { return 1; }
    // CHECK:STDERR:                          ^
    // CHECK:STDERR: fail_missing_guard_open_paren.carbon:[[@LINE+4]]:21: note: possibly missing `(` here [PossiblyMissingBracketHere]
    // CHECK:STDERR:     case x: bool if false) => { return 1; }
    // CHECK:STDERR:                     ^~~~~
    // CHECK:STDERR:
    case x: bool if false) => { return 1; }
  }
  return 0;
}

// CHECK:STDOUT: - filename: fail_missing_guard_open_paren.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameMaybeBeforeSignature', text: 'f'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'ReturnType', text: '->', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 7},
// CHECK:STDOUT:           {kind: 'MatchIntroducer', text: 'match'},
// CHECK:STDOUT:             {kind: 'MatchConditionStart', text: '('},
// CHECK:STDOUT:             {kind: 'BoolLiteralTrue', text: 'true'},
// CHECK:STDOUT:           {kind: 'MatchCondition', text: ')', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'MatchStatementStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:               {kind: 'MatchCaseIntroducer', text: 'case'},
// CHECK:STDOUT:                 {kind: 'IdentifierNameNotBeforeSignature', text: 'x'},
// CHECK:STDOUT:                 {kind: 'BoolTypeLiteral', text: 'bool'},
// CHECK:STDOUT:               {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:                 {kind: 'MatchCaseGuardIntroducer', text: 'if'},
// CHECK:STDOUT:                 {kind: 'MatchCaseGuardStart', text: '('},
// CHECK:STDOUT:                 {kind: 'BoolLiteralFalse', text: 'false'},
// CHECK:STDOUT:               {kind: 'MatchCaseGuard', text: ')', subtree_size: 4},
// CHECK:STDOUT:             {kind: 'MatchCase', text: '=>', subtree_size: 9},
// CHECK:STDOUT:           {kind: 'MatchHandlerStart', text: '{', subtree_size: 10},
// CHECK:STDOUT:             {kind: 'ReturnStatementStart', text: 'return'},
// CHECK:STDOUT:             {kind: 'IntLiteral', text: '1'},
// CHECK:STDOUT:           {kind: 'ReturnStatement', text: ';', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'MatchHandler', text: '}', subtree_size: 14},
// CHECK:STDOUT:       {kind: 'MatchStatement', text: '}', subtree_size: 20},
// CHECK:STDOUT:         {kind: 'ReturnStatementStart', text: 'return'},
// CHECK:STDOUT:         {kind: 'IntLiteral', text: '0'},
// CHECK:STDOUT:       {kind: 'ReturnStatement', text: ';', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 31},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
