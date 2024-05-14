// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn f() -> i32 {
  match (3) {
    // CHECK:STDERR: fail_missing_case_pattern.carbon:[[@LINE+3]]:10: ERROR: Expected binding pattern.
    // CHECK:STDERR:     case => { return 2; }
    // CHECK:STDERR:          ^~
    case => { return 2; }
  }
  return 0;
}

// CHECK:STDOUT: - filename: fail_missing_case_pattern.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'f'},
// CHECK:STDOUT:           {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:         {kind: 'TuplePattern', text: ')', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:         {kind: 'ReturnType', text: '->', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 7},
// CHECK:STDOUT:           {kind: 'MatchIntroducer', text: 'match'},
// CHECK:STDOUT:             {kind: 'MatchConditionStart', text: '('},
// CHECK:STDOUT:             {kind: 'IntLiteral', text: '3'},
// CHECK:STDOUT:           {kind: 'MatchCondition', text: ')', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'MatchStatementStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:             {kind: 'MatchCaseIntroducer', text: 'case'},
// CHECK:STDOUT:               {kind: 'IdentifierName', text: '=>', has_error: yes},
// CHECK:STDOUT:               {kind: 'InvalidParse', text: '=>', has_error: yes},
// CHECK:STDOUT:             {kind: 'BindingPattern', text: '=>', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:           {kind: 'MatchCaseStart', text: '=>', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:         {kind: 'MatchCase', text: '=>', has_error: yes, subtree_size: 6},
// CHECK:STDOUT:       {kind: 'MatchStatement', text: '}', subtree_size: 12},
// CHECK:STDOUT:         {kind: 'ReturnStatementStart', text: 'return'},
// CHECK:STDOUT:         {kind: 'IntLiteral', text: '0'},
// CHECK:STDOUT:       {kind: 'ReturnStatement', text: ';', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 23},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
