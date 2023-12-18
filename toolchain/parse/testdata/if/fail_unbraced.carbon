// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// TODO: This should have an error.
// AUTOUPDATE

fn F() {
  if (a)
    // CHECK:STDERR: fail_unbraced.carbon:[[@LINE+3]]:5: ERROR: Expected braced code block.
    // CHECK:STDERR:     if (b)
    // CHECK:STDERR:     ^~
    if (b)
      // CHECK:STDERR: fail_unbraced.carbon:[[@LINE+3]]:7: ERROR: Expected braced code block.
      // CHECK:STDERR:       if (c)
      // CHECK:STDERR:       ^~
      if (c)
        // CHECK:STDERR: fail_unbraced.carbon:[[@LINE+3]]:9: ERROR: Expected braced code block.
        // CHECK:STDERR:         d;
        // CHECK:STDERR:         ^
        d;
}

// CHECK:STDOUT: - filename: fail_unbraced.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'F'},
// CHECK:STDOUT:           {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:         {kind: 'TuplePattern', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'IfConditionStart', text: '('},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'a'},
// CHECK:STDOUT:         {kind: 'IfCondition', text: ')', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'CodeBlockStart', text: 'if', has_error: yes},
// CHECK:STDOUT:               {kind: 'IfConditionStart', text: '('},
// CHECK:STDOUT:               {kind: 'IdentifierNameExpr', text: 'b'},
// CHECK:STDOUT:             {kind: 'IfCondition', text: ')', subtree_size: 3},
// CHECK:STDOUT:               {kind: 'CodeBlockStart', text: 'if', has_error: yes},
// CHECK:STDOUT:                   {kind: 'IfConditionStart', text: '('},
// CHECK:STDOUT:                   {kind: 'IdentifierNameExpr', text: 'c'},
// CHECK:STDOUT:                 {kind: 'IfCondition', text: ')', subtree_size: 3},
// CHECK:STDOUT:                   {kind: 'CodeBlockStart', text: 'd', has_error: yes},
// CHECK:STDOUT:                     {kind: 'IdentifierNameExpr', text: 'd'},
// CHECK:STDOUT:                   {kind: 'ExprStatement', text: ';', subtree_size: 2},
// CHECK:STDOUT:                 {kind: 'CodeBlock', text: 'd', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:               {kind: 'IfStatement', text: 'if', subtree_size: 8},
// CHECK:STDOUT:             {kind: 'CodeBlock', text: 'if', has_error: yes, subtree_size: 10},
// CHECK:STDOUT:           {kind: 'IfStatement', text: 'if', subtree_size: 14},
// CHECK:STDOUT:         {kind: 'CodeBlock', text: 'if', has_error: yes, subtree_size: 16},
// CHECK:STDOUT:       {kind: 'IfStatement', text: 'if', subtree_size: 20},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 26},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
