// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn f() -> i32 {
  match (3) {
    // CHECK:STDERR: fail_missing_default_arrow.carbon:[[@LINE+3]]:13: ERROR: Expected `=>` introducing statement block.
    // CHECK:STDERR:     default { return 1; }
    // CHECK:STDERR:             ^
    default { return 1; }
  }
  return 0;
}

// CHECK:STDOUT: - filename: fail_missing_default_arrow.carbon
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
// CHECK:STDOUT:             {kind: 'MatchDefaultIntroducer', text: 'default'},
// CHECK:STDOUT:             {kind: 'MatchDefaultEqualGreater', text: '{', has_error: yes},
// CHECK:STDOUT:           {kind: 'MatchDefaultStart', text: '{', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:         {kind: 'MatchDefault', text: '{', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:       {kind: 'MatchStatement', text: '}', subtree_size: 10},
// CHECK:STDOUT:         {kind: 'ReturnStatementStart', text: 'return'},
// CHECK:STDOUT:         {kind: 'IntLiteral', text: '0'},
// CHECK:STDOUT:       {kind: 'ReturnStatement', text: ';', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 21},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
