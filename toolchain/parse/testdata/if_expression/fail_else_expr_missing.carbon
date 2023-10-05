// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() {
  // CHECK:STDERR: fail_else_expr_missing.carbon:[[@LINE+3]]:35: ERROR: Expected expression.
  // CHECK:STDERR:   var n: i32 = if true then 1 else;
  // CHECK:STDERR:                                   ^
  var n: i32 = if true then 1 else;
}

// CHECK:STDOUT: - filename: fail_else_expr_missing.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'Name', text: 'F'},
// CHECK:STDOUT:           {kind: 'ParameterListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ParameterList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:           {kind: 'Name', text: 'n'},
// CHECK:STDOUT:           {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:         {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:             {kind: 'Literal', text: 'true'},
// CHECK:STDOUT:           {kind: 'IfExpressionIf', text: 'if', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'Literal', text: '1'},
// CHECK:STDOUT:           {kind: 'IfExpressionThen', text: 'then', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'InvalidParse', text: ';', has_error: yes},
// CHECK:STDOUT:         {kind: 'IfExpressionElse', text: 'else', has_error: yes, subtree_size: 6},
// CHECK:STDOUT:       {kind: 'VariableDeclaration', text: ';', subtree_size: 12},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 18},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
