// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/match/fail_missing_matched_expr.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/match/fail_missing_matched_expr.carbon

fn f() -> i32 {
  // CHECK:STDERR: fail_missing_matched_expr.carbon:[[@LINE+4]]:9: error: expected `(` after `match` [ExpectedParenAfter]
  // CHECK:STDERR:   match {
  // CHECK:STDERR:         ^
  // CHECK:STDERR:
  match {
    default => { return 1; }
  }
  return 0;
}

// CHECK:STDOUT: - filename: fail_missing_matched_expr.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'f'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'ReturnType', text: '->', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 7},
// CHECK:STDOUT:           {kind: 'MatchIntroducer', text: 'match'},
// CHECK:STDOUT:             {kind: 'MatchConditionStart', text: 'match', has_error: yes},
// CHECK:STDOUT:             {kind: 'InvalidParse', text: '{', has_error: yes},
// CHECK:STDOUT:           {kind: 'MatchCondition', text: 'match', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:         {kind: 'MatchStatementStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:             {kind: 'MatchDefaultIntroducer', text: 'default'},
// CHECK:STDOUT:             {kind: 'MatchDefaultEqualGreater', text: '=>'},
// CHECK:STDOUT:           {kind: 'MatchDefaultStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:             {kind: 'ReturnStatementStart', text: 'return'},
// CHECK:STDOUT:             {kind: 'IntLiteral', text: '1'},
// CHECK:STDOUT:           {kind: 'ReturnStatement', text: ';', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'MatchDefault', text: '}', subtree_size: 7},
// CHECK:STDOUT:       {kind: 'MatchStatement', text: '}', subtree_size: 13},
// CHECK:STDOUT:         {kind: 'ReturnStatementStart', text: 'return'},
// CHECK:STDOUT:         {kind: 'IntLiteral', text: '0'},
// CHECK:STDOUT:       {kind: 'ReturnStatement', text: ';', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 24},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
