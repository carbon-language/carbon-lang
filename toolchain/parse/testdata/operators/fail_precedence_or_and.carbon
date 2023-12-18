// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() {
  // CHECK:STDERR: fail_precedence_or_and.carbon:[[@LINE+3]]:10: ERROR: Parentheses are required to disambiguate operator precedence.
  // CHECK:STDERR:   a or b and c;
  // CHECK:STDERR:          ^~~
  a or b and c;
}

// CHECK:STDOUT: - filename: fail_precedence_or_and.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'F'},
// CHECK:STDOUT:           {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:         {kind: 'TuplePattern', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:                 {kind: 'IdentifierNameExpr', text: 'a'},
// CHECK:STDOUT:               {kind: 'ShortCircuitOperandOr', text: 'or', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'IdentifierNameExpr', text: 'b'},
// CHECK:STDOUT:             {kind: 'ShortCircuitOperatorOr', text: 'or', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'ShortCircuitOperandAnd', text: 'and', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'c'},
// CHECK:STDOUT:         {kind: 'ShortCircuitOperatorAnd', text: 'and', has_error: yes, subtree_size: 7},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 8},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 14},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
