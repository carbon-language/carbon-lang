// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn foo() {
  // CHECK:STDERR: fail_missing_in.carbon:[[@LINE+3]]:19: ERROR: Expected `in` after loop `var` declaration.
  // CHECK:STDERR:   for (var x: i32 y) {
  // CHECK:STDERR:                   ^
  for (var x: i32 y) {
    Print(x);
  }
}

// CHECK:STDOUT: - filename: fail_missing_in.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'Name', text: 'foo'},
// CHECK:STDOUT:           {kind: 'ParameterListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ParameterList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'ForHeaderStart', text: '('},
// CHECK:STDOUT:             {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:               {kind: 'Name', text: 'x'},
// CHECK:STDOUT:               {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:             {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'ForIn', text: 'var', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:           {kind: 'NameExpression', text: 'y'},
// CHECK:STDOUT:         {kind: 'ForHeader', text: ')', subtree_size: 8},
// CHECK:STDOUT:           {kind: 'CodeBlockStart', text: '{'},
// CHECK:STDOUT:                 {kind: 'NameExpression', text: 'Print'},
// CHECK:STDOUT:               {kind: 'CallExpressionStart', text: '(', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'NameExpression', text: 'x'},
// CHECK:STDOUT:             {kind: 'CallExpression', text: ')', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'ExpressionStatement', text: ';', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'CodeBlock', text: '}', subtree_size: 7},
// CHECK:STDOUT:       {kind: 'ForStatement', text: 'for', subtree_size: 16},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 22},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
