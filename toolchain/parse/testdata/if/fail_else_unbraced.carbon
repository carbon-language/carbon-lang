// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() {
  if (a)
    // CHECK:STDERR: fail_else_unbraced.carbon:[[@LINE+3]]:5: ERROR: Expected braced code block.
    // CHECK:STDERR:     if (b)
    // CHECK:STDERR:     ^
    if (b)
      // CHECK:STDERR: fail_else_unbraced.carbon:[[@LINE+3]]:7: ERROR: Expected braced code block.
      // CHECK:STDERR:       c;
      // CHECK:STDERR:       ^
      c;
    else
      // CHECK:STDERR: fail_else_unbraced.carbon:[[@LINE+3]]:7: ERROR: Expected braced code block.
      // CHECK:STDERR:       d;
      // CHECK:STDERR:       ^
      d;
  else
    // CHECK:STDERR: fail_else_unbraced.carbon:[[@LINE+3]]:5: ERROR: Expected braced code block.
    // CHECK:STDERR:     e;
    // CHECK:STDERR:     ^
    e;
  if (x) { f; }
  else if (x) { g; }
  else { h; }
}

// CHECK:STDOUT: - filename: fail_else_unbraced.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'Name', text: 'F'},
// CHECK:STDOUT:           {kind: 'ParameterListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ParameterList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'IfConditionStart', text: '('},
// CHECK:STDOUT:           {kind: 'NameExpression', text: 'a'},
// CHECK:STDOUT:         {kind: 'IfCondition', text: ')', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'CodeBlockStart', text: 'if', has_error: yes},
// CHECK:STDOUT:               {kind: 'IfConditionStart', text: '('},
// CHECK:STDOUT:               {kind: 'NameExpression', text: 'b'},
// CHECK:STDOUT:             {kind: 'IfCondition', text: ')', subtree_size: 3},
// CHECK:STDOUT:               {kind: 'CodeBlockStart', text: 'c', has_error: yes},
// CHECK:STDOUT:                 {kind: 'NameExpression', text: 'c'},
// CHECK:STDOUT:               {kind: 'ExpressionStatement', text: ';', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'CodeBlock', text: 'c', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:             {kind: 'IfStatementElse', text: 'else'},
// CHECK:STDOUT:               {kind: 'CodeBlockStart', text: 'd', has_error: yes},
// CHECK:STDOUT:                 {kind: 'NameExpression', text: 'd'},
// CHECK:STDOUT:               {kind: 'ExpressionStatement', text: ';', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'CodeBlock', text: 'd', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:           {kind: 'IfStatement', text: 'if', subtree_size: 13},
// CHECK:STDOUT:         {kind: 'CodeBlock', text: 'if', has_error: yes, subtree_size: 15},
// CHECK:STDOUT:         {kind: 'IfStatementElse', text: 'else'},
// CHECK:STDOUT:           {kind: 'CodeBlockStart', text: 'e', has_error: yes},
// CHECK:STDOUT:             {kind: 'NameExpression', text: 'e'},
// CHECK:STDOUT:           {kind: 'ExpressionStatement', text: ';', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'CodeBlock', text: 'e', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:       {kind: 'IfStatement', text: 'if', subtree_size: 24},
// CHECK:STDOUT:           {kind: 'IfConditionStart', text: '('},
// CHECK:STDOUT:           {kind: 'NameExpression', text: 'x'},
// CHECK:STDOUT:         {kind: 'IfCondition', text: ')', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'CodeBlockStart', text: '{'},
// CHECK:STDOUT:             {kind: 'NameExpression', text: 'f'},
// CHECK:STDOUT:           {kind: 'ExpressionStatement', text: ';', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'CodeBlock', text: '}', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'IfStatementElse', text: 'else'},
// CHECK:STDOUT:             {kind: 'IfConditionStart', text: '('},
// CHECK:STDOUT:             {kind: 'NameExpression', text: 'x'},
// CHECK:STDOUT:           {kind: 'IfCondition', text: ')', subtree_size: 3},
// CHECK:STDOUT:             {kind: 'CodeBlockStart', text: '{'},
// CHECK:STDOUT:               {kind: 'NameExpression', text: 'g'},
// CHECK:STDOUT:             {kind: 'ExpressionStatement', text: ';', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'CodeBlock', text: '}', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'IfStatementElse', text: 'else'},
// CHECK:STDOUT:             {kind: 'CodeBlockStart', text: '{'},
// CHECK:STDOUT:               {kind: 'NameExpression', text: 'h'},
// CHECK:STDOUT:             {kind: 'ExpressionStatement', text: ';', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'CodeBlock', text: '}', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'IfStatement', text: 'if', subtree_size: 13},
// CHECK:STDOUT:       {kind: 'IfStatement', text: 'if', subtree_size: 22},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 52},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
