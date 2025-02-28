// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/auto/match_case.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/auto/match_case.carbon

fn f() -> bool {
  match (true) {
    case x: auto if (x == true) => { return true; }
    default => { return false; }
  }
}

// CHECK:STDOUT: - filename: match_case.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'f'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'BoolTypeLiteral', text: 'bool'},
// CHECK:STDOUT:         {kind: 'ReturnType', text: '->', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 7},
// CHECK:STDOUT:           {kind: 'MatchIntroducer', text: 'match'},
// CHECK:STDOUT:             {kind: 'MatchConditionStart', text: '('},
// CHECK:STDOUT:             {kind: 'BoolLiteralTrue', text: 'true'},
// CHECK:STDOUT:           {kind: 'MatchCondition', text: ')', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'MatchStatementStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:             {kind: 'MatchCaseIntroducer', text: 'case'},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeParams', text: 'x'},
// CHECK:STDOUT:               {kind: 'AutoTypeLiteral', text: 'auto'},
// CHECK:STDOUT:             {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:               {kind: 'MatchCaseGuardIntroducer', text: 'if'},
// CHECK:STDOUT:               {kind: 'MatchCaseGuardStart', text: '('},
// CHECK:STDOUT:                 {kind: 'IdentifierNameExpr', text: 'x'},
// CHECK:STDOUT:                 {kind: 'BoolLiteralTrue', text: 'true'},
// CHECK:STDOUT:               {kind: 'InfixOperatorEqualEqual', text: '==', subtree_size: 3},
// CHECK:STDOUT:             {kind: 'MatchCaseGuard', text: ')', subtree_size: 6},
// CHECK:STDOUT:             {kind: 'MatchCaseEqualGreater', text: '=>'},
// CHECK:STDOUT:           {kind: 'MatchCaseStart', text: '{', subtree_size: 12},
// CHECK:STDOUT:             {kind: 'ReturnStatementStart', text: 'return'},
// CHECK:STDOUT:             {kind: 'BoolLiteralTrue', text: 'true'},
// CHECK:STDOUT:           {kind: 'ReturnStatement', text: ';', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'MatchCase', text: '}', subtree_size: 16},
// CHECK:STDOUT:             {kind: 'MatchDefaultIntroducer', text: 'default'},
// CHECK:STDOUT:             {kind: 'MatchDefaultEqualGreater', text: '=>'},
// CHECK:STDOUT:           {kind: 'MatchDefaultStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:             {kind: 'ReturnStatementStart', text: 'return'},
// CHECK:STDOUT:             {kind: 'BoolLiteralFalse', text: 'false'},
// CHECK:STDOUT:           {kind: 'ReturnStatement', text: ';', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'MatchDefault', text: '}', subtree_size: 7},
// CHECK:STDOUT:       {kind: 'MatchStatement', text: '}', subtree_size: 29},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 37},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
