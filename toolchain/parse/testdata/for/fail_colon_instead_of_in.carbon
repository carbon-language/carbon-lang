// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn foo() {
  // CHECK:STDERR: fail_colon_instead_of_in.carbon:[[@LINE+3]]:19: ERROR: `:` should be replaced by `in`.
  // CHECK:STDERR:   for (var x: i32 : y) {
  // CHECK:STDERR:                   ^
  for (var x: i32 : y) {
    Print(x);
  }
}

// CHECK:STDOUT: - filename: fail_colon_instead_of_in.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'foo'},
// CHECK:STDOUT:           {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:         {kind: 'TuplePattern', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'ForHeaderStart', text: '('},
// CHECK:STDOUT:             {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:               {kind: 'IdentifierName', text: 'x'},
// CHECK:STDOUT:               {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:             {kind: 'BindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'ForIn', text: ':', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'y'},
// CHECK:STDOUT:         {kind: 'ForHeader', text: ')', subtree_size: 8},
// CHECK:STDOUT:           {kind: 'CodeBlockStart', text: '{'},
// CHECK:STDOUT:                 {kind: 'IdentifierNameExpr', text: 'Print'},
// CHECK:STDOUT:               {kind: 'CallExprStart', text: '(', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'IdentifierNameExpr', text: 'x'},
// CHECK:STDOUT:             {kind: 'CallExpr', text: ')', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'ExprStatement', text: ';', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'CodeBlock', text: '}', subtree_size: 7},
// CHECK:STDOUT:       {kind: 'ForStatement', text: 'for', subtree_size: 16},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 22},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
