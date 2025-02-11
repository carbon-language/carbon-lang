// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/match/fail_unexpected_tokens_in_cases_block.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/match/fail_unexpected_tokens_in_cases_block.carbon

fn f() -> i32 {
  var x: i32 = 3;
  match (x) {
    // CHECK:STDERR: fail_unexpected_tokens_in_cases_block.carbon:[[@LINE+4]]:5: error: unexpected `let`; expected `case`, `default` or `}` [UnexpectedTokenInMatchCasesBlock]
    // CHECK:STDERR:     let u: i32 =
    // CHECK:STDERR:     ^~~
    // CHECK:STDERR:
    let u: i32 =
    case y: i32 if (y > 2) => { return 1; }
    // CHECK:STDERR: fail_unexpected_tokens_in_cases_block.carbon:[[@LINE+4]]:5: error: unexpected `if`; expected `case`, `default` or `}` [UnexpectedTokenInMatchCasesBlock]
    // CHECK:STDERR:     if (true) {
    // CHECK:STDERR:     ^~
    // CHECK:STDERR:
    if (true) {
      return 2;
    }
    // CHECK:STDERR: fail_unexpected_tokens_in_cases_block.carbon:[[@LINE+4]]:5: error: unexpected `var`; expected `case`, `default` or `}` [UnexpectedTokenInMatchCasesBlock]
    // CHECK:STDERR:     var w
    // CHECK:STDERR:     ^~~
    // CHECK:STDERR:
    var w
    case z: i32 if (z == 3) => { return 3; }
    // CHECK:STDERR: fail_unexpected_tokens_in_cases_block.carbon:[[@LINE+4]]:5: error: unexpected `{`; expected `case`, `default` or `}` [UnexpectedTokenInMatchCasesBlock]
    // CHECK:STDERR:     {
    // CHECK:STDERR:     ^
    // CHECK:STDERR:
    {
      break;
    }
    default => { return 3; }
    // CHECK:STDERR: fail_unexpected_tokens_in_cases_block.carbon:[[@LINE+4]]:5: error: unexpected `return`; expected `case`, `default` or `}` [UnexpectedTokenInMatchCasesBlock]
    // CHECK:STDERR:     return 1;
    // CHECK:STDERR:     ^~~~~~
    // CHECK:STDERR:
    return 1;
    // CHECK:STDERR: fail_unexpected_tokens_in_cases_block.carbon:[[@LINE+4]]:5: error: unexpected `(`; expected `case`, `default` or `}` [UnexpectedTokenInMatchCasesBlock]
    // CHECK:STDERR:     ((), (), ())
    // CHECK:STDERR:     ^
    // CHECK:STDERR:
    ((), (), ())
  }
  return 0;
}

// CHECK:STDOUT: - filename: fail_unexpected_tokens_in_cases_block.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierNameBeforeParams', text: 'f'},
// CHECK:STDOUT:           {kind: 'ExplicitParamListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ExplicitParamList', text: ')', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'ReturnType', text: '->', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 7},
// CHECK:STDOUT:         {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:             {kind: 'IdentifierNameNotBeforeParams', text: 'x'},
// CHECK:STDOUT:             {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'VarBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'VariablePattern', text: 'var', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'IntLiteral', text: '3'},
// CHECK:STDOUT:       {kind: 'VariableDecl', text: ';', subtree_size: 8},
// CHECK:STDOUT:           {kind: 'MatchIntroducer', text: 'match'},
// CHECK:STDOUT:             {kind: 'MatchConditionStart', text: '('},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'x'},
// CHECK:STDOUT:           {kind: 'MatchCondition', text: ')', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'MatchStatementStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:             {kind: 'MatchCaseIntroducer', text: 'case'},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeParams', text: 'y'},
// CHECK:STDOUT:               {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:             {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:               {kind: 'MatchCaseGuardIntroducer', text: 'if'},
// CHECK:STDOUT:               {kind: 'MatchCaseGuardStart', text: '('},
// CHECK:STDOUT:                 {kind: 'IdentifierNameExpr', text: 'y'},
// CHECK:STDOUT:                 {kind: 'IntLiteral', text: '2'},
// CHECK:STDOUT:               {kind: 'InfixOperatorGreater', text: '>', subtree_size: 3},
// CHECK:STDOUT:             {kind: 'MatchCaseGuard', text: ')', subtree_size: 6},
// CHECK:STDOUT:             {kind: 'MatchCaseEqualGreater', text: '=>'},
// CHECK:STDOUT:           {kind: 'MatchCaseStart', text: '{', subtree_size: 12},
// CHECK:STDOUT:             {kind: 'ReturnStatementStart', text: 'return'},
// CHECK:STDOUT:             {kind: 'IntLiteral', text: '1'},
// CHECK:STDOUT:           {kind: 'ReturnStatement', text: ';', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'MatchCase', text: '}', subtree_size: 16},
// CHECK:STDOUT:             {kind: 'MatchCaseIntroducer', text: 'case'},
// CHECK:STDOUT:               {kind: 'IdentifierNameNotBeforeParams', text: 'z'},
// CHECK:STDOUT:               {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:             {kind: 'LetBindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:               {kind: 'MatchCaseGuardIntroducer', text: 'if'},
// CHECK:STDOUT:               {kind: 'MatchCaseGuardStart', text: '('},
// CHECK:STDOUT:                 {kind: 'IdentifierNameExpr', text: 'z'},
// CHECK:STDOUT:                 {kind: 'IntLiteral', text: '3'},
// CHECK:STDOUT:               {kind: 'InfixOperatorEqualEqual', text: '==', subtree_size: 3},
// CHECK:STDOUT:             {kind: 'MatchCaseGuard', text: ')', subtree_size: 6},
// CHECK:STDOUT:             {kind: 'MatchCaseEqualGreater', text: '=>'},
// CHECK:STDOUT:           {kind: 'MatchCaseStart', text: '{', subtree_size: 12},
// CHECK:STDOUT:             {kind: 'ReturnStatementStart', text: 'return'},
// CHECK:STDOUT:             {kind: 'IntLiteral', text: '3'},
// CHECK:STDOUT:           {kind: 'ReturnStatement', text: ';', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'MatchCase', text: '}', subtree_size: 16},
// CHECK:STDOUT:             {kind: 'MatchDefaultIntroducer', text: 'default'},
// CHECK:STDOUT:             {kind: 'MatchDefaultEqualGreater', text: '=>'},
// CHECK:STDOUT:           {kind: 'MatchDefaultStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:             {kind: 'ReturnStatementStart', text: 'return'},
// CHECK:STDOUT:             {kind: 'IntLiteral', text: '3'},
// CHECK:STDOUT:           {kind: 'ReturnStatement', text: ';', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'MatchDefault', text: '}', subtree_size: 7},
// CHECK:STDOUT:       {kind: 'MatchStatement', text: '}', has_error: yes, subtree_size: 45},
// CHECK:STDOUT:         {kind: 'ReturnStatementStart', text: 'return'},
// CHECK:STDOUT:         {kind: 'IntLiteral', text: '0'},
// CHECK:STDOUT:       {kind: 'ReturnStatement', text: ';', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 64},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
