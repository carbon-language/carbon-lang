// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() {
  // CHECK:STDERR: fail_errors.carbon:[[@LINE+3]]:6: ERROR: Expected `(` after `if`.
  // CHECK:STDERR:   if a {}
  // CHECK:STDERR:      ^
  if a {}
  // CHECK:STDERR: fail_errors.carbon:[[@LINE+3]]:7: ERROR: Expected expression.
  // CHECK:STDERR:   if () {}
  // CHECK:STDERR:       ^
  if () {}
  // CHECK:STDERR: fail_errors.carbon:[[@LINE+3]]:9: ERROR: Unexpected tokens before `)`.
  // CHECK:STDERR:   if (b c) {}
  // CHECK:STDERR:         ^
  if (b c) {}
  if (d)
// CHECK:STDERR: fail_errors.carbon:[[@LINE+6]]:1: ERROR: Expected braced code block.
// CHECK:STDERR: }
// CHECK:STDERR: ^
// CHECK:STDERR: fail_errors.carbon:[[@LINE+3]]:1: ERROR: Expected expression.
// CHECK:STDERR: }
// CHECK:STDERR: ^
}

// CHECK:STDOUT: - filename: fail_errors.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'F'},
// CHECK:STDOUT:           {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:         {kind: 'TuplePattern', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'IfConditionStart', text: 'if', has_error: yes},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'a'},
// CHECK:STDOUT:         {kind: 'IfCondition', text: 'if', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:           {kind: 'CodeBlockStart', text: '{'},
// CHECK:STDOUT:         {kind: 'CodeBlock', text: '}', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'IfStatement', text: 'if', subtree_size: 6},
// CHECK:STDOUT:           {kind: 'IfConditionStart', text: '('},
// CHECK:STDOUT:           {kind: 'InvalidParse', text: ')', has_error: yes},
// CHECK:STDOUT:         {kind: 'IfCondition', text: ')', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:           {kind: 'CodeBlockStart', text: '{'},
// CHECK:STDOUT:         {kind: 'CodeBlock', text: '}', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'IfStatement', text: 'if', subtree_size: 6},
// CHECK:STDOUT:           {kind: 'IfConditionStart', text: '('},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'b'},
// CHECK:STDOUT:         {kind: 'IfCondition', text: ')', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:           {kind: 'CodeBlockStart', text: '{'},
// CHECK:STDOUT:         {kind: 'CodeBlock', text: '}', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'IfStatement', text: 'if', subtree_size: 6},
// CHECK:STDOUT:           {kind: 'IfConditionStart', text: '('},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'd'},
// CHECK:STDOUT:         {kind: 'IfCondition', text: ')', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'CodeBlockStart', text: '}', has_error: yes},
// CHECK:STDOUT:           {kind: 'InvalidParse', text: '}', has_error: yes},
// CHECK:STDOUT:         {kind: 'CodeBlock', text: '}', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:       {kind: 'IfStatement', text: 'if', subtree_size: 7},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 31},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
