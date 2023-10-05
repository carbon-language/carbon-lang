// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() {
  // CHECK:STDERR: fail_square_brackets.carbon:[[@LINE+9]]:6: ERROR: Expected `(` after `if`.
  // CHECK:STDERR:   if [] {}
  // CHECK:STDERR:      ^
  // CHECK:STDERR: fail_square_brackets.carbon:[[@LINE+6]]:7: ERROR: Expected expression.
  // CHECK:STDERR:   if [] {}
  // CHECK:STDERR:       ^
  // CHECK:STDERR: fail_square_brackets.carbon:[[@LINE+3]]:7: ERROR: Expected `;` in array type.
  // CHECK:STDERR:   if [] {}
  // CHECK:STDERR:       ^
  if [] {}
}

// CHECK:STDOUT: - filename: fail_square_brackets.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'Name', text: 'F'},
// CHECK:STDOUT:           {kind: 'ParameterListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ParameterList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'IfConditionStart', text: 'if', has_error: yes},
// CHECK:STDOUT:               {kind: 'ArrayExpressionStart', text: '['},
// CHECK:STDOUT:               {kind: 'InvalidParse', text: ']', has_error: yes},
// CHECK:STDOUT:             {kind: 'ArrayExpressionSemi', text: ']', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:           {kind: 'ArrayExpression', text: ']', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:         {kind: 'IfCondition', text: 'if', has_error: yes, subtree_size: 6},
// CHECK:STDOUT:           {kind: 'CodeBlockStart', text: '{'},
// CHECK:STDOUT:         {kind: 'CodeBlock', text: '}', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'IfStatement', text: 'if', subtree_size: 9},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 15},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
