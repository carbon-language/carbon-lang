// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() {
  // CHECK:STDERR: fail_missing_cond.carbon:[[@LINE+6]]:7: ERROR: Expected `(` after `for`.
  // CHECK:STDERR:   for {
  // CHECK:STDERR:       ^
  // CHECK:STDERR: fail_missing_cond.carbon:[[@LINE+3]]:7: ERROR: Expected `var` declaration.
  // CHECK:STDERR:   for {
  // CHECK:STDERR:       ^
  for {
  }
// CHECK:STDERR: fail_missing_cond.carbon:[[@LINE+6]]:1: ERROR: Expected braced code block.
// CHECK:STDERR: }
// CHECK:STDERR: ^
// CHECK:STDERR: fail_missing_cond.carbon:[[@LINE+3]]:1: ERROR: Expected expression.
// CHECK:STDERR: }
// CHECK:STDERR: ^
}

// CHECK:STDOUT: - filename: fail_missing_cond.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'F'},
// CHECK:STDOUT:           {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:         {kind: 'TuplePattern', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'ForHeaderStart', text: 'for', has_error: yes},
// CHECK:STDOUT:             {kind: 'StructLiteralOrStructTypeLiteralStart', text: '{'},
// CHECK:STDOUT:           {kind: 'StructLiteral', text: '}', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'ForHeader', text: 'for', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:           {kind: 'CodeBlockStart', text: '}', has_error: yes},
// CHECK:STDOUT:           {kind: 'InvalidParse', text: '}', has_error: yes},
// CHECK:STDOUT:         {kind: 'CodeBlock', text: '}', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ForStatement', text: 'for', subtree_size: 8},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 14},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
