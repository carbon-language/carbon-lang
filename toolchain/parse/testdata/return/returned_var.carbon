// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> String {
  returned var s: String = "hello";
  // CHECK:STDERR: returned_var.carbon:[[@LINE+6]]:10: ERROR: Expected expression.
  // CHECK:STDERR:   return var;
  // CHECK:STDERR:          ^
  // CHECK:STDERR: returned_var.carbon:[[@LINE+3]]:10: ERROR: `return` statements must end with a `;`.
  // CHECK:STDERR:   return var;
  // CHECK:STDERR:          ^
  return var;
}

// CHECK:STDOUT: - filename: returned_var.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'Name', text: 'F'},
// CHECK:STDOUT:           {kind: 'ParameterListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ParameterList', text: ')', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'Literal', text: 'String'},
// CHECK:STDOUT:         {kind: 'ReturnType', text: '->', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 7},
// CHECK:STDOUT:         {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:         {kind: 'ReturnedSpecifier', text: 'returned'},
// CHECK:STDOUT:           {kind: 'Name', text: 's'},
// CHECK:STDOUT:           {kind: 'Literal', text: 'String'},
// CHECK:STDOUT:         {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'Literal', text: '"hello"'},
// CHECK:STDOUT:       {kind: 'VariableDeclaration', text: ';', subtree_size: 8},
// CHECK:STDOUT:         {kind: 'ReturnStatementStart', text: 'return'},
// CHECK:STDOUT:         {kind: 'InvalidParse', text: 'var', has_error: yes},
// CHECK:STDOUT:       {kind: 'ReturnStatement', text: ';', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 19},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
