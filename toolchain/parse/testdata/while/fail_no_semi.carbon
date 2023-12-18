// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() {
  while (a) {
    if (b) {
      break
    // CHECK:STDERR: fail_no_semi.carbon:[[@LINE+3]]:5: ERROR: `break` statements must end with a `;`.
    // CHECK:STDERR:     }
    // CHECK:STDERR:     ^
    }
    if (c) {
      continue
    // CHECK:STDERR: fail_no_semi.carbon:[[@LINE+3]]:5: ERROR: `continue` statements must end with a `;`.
    // CHECK:STDERR:     }
    // CHECK:STDERR:     ^
    }
  }
}

// CHECK:STDOUT: - filename: fail_no_semi.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'F'},
// CHECK:STDOUT:           {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:         {kind: 'TuplePattern', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'WhileConditionStart', text: '('},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'a'},
// CHECK:STDOUT:         {kind: 'WhileCondition', text: ')', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'CodeBlockStart', text: '{'},
// CHECK:STDOUT:               {kind: 'IfConditionStart', text: '('},
// CHECK:STDOUT:               {kind: 'IdentifierNameExpr', text: 'b'},
// CHECK:STDOUT:             {kind: 'IfCondition', text: ')', subtree_size: 3},
// CHECK:STDOUT:               {kind: 'CodeBlockStart', text: '{'},
// CHECK:STDOUT:                 {kind: 'BreakStatementStart', text: 'break'},
// CHECK:STDOUT:               {kind: 'BreakStatement', text: 'break', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:             {kind: 'CodeBlock', text: '}', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'IfStatement', text: 'if', subtree_size: 8},
// CHECK:STDOUT:               {kind: 'IfConditionStart', text: '('},
// CHECK:STDOUT:               {kind: 'IdentifierNameExpr', text: 'c'},
// CHECK:STDOUT:             {kind: 'IfCondition', text: ')', subtree_size: 3},
// CHECK:STDOUT:               {kind: 'CodeBlockStart', text: '{'},
// CHECK:STDOUT:                 {kind: 'ContinueStatementStart', text: 'continue'},
// CHECK:STDOUT:               {kind: 'ContinueStatement', text: 'continue', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:             {kind: 'CodeBlock', text: '}', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'IfStatement', text: 'if', subtree_size: 8},
// CHECK:STDOUT:         {kind: 'CodeBlock', text: '}', subtree_size: 18},
// CHECK:STDOUT:       {kind: 'WhileStatement', text: 'while', subtree_size: 22},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 28},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
