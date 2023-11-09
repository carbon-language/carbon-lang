// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() {
  a and b and c;
}

// CHECK:STDOUT: - filename: associative.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'Name', text: 'F'},
// CHECK:STDOUT:           {kind: 'ParameterListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ParameterList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:                 {kind: 'NameExpr', text: 'a'},
// CHECK:STDOUT:               {kind: 'ShortCircuitOperand', text: 'and', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'NameExpr', text: 'b'},
// CHECK:STDOUT:             {kind: 'InfixOperator', text: 'and', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'ShortCircuitOperand', text: 'and', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'NameExpr', text: 'c'},
// CHECK:STDOUT:         {kind: 'InfixOperator', text: 'and', subtree_size: 7},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 8},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 14},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
