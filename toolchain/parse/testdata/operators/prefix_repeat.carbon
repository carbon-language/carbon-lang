// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() {
  ***&n;
}

// CHECK:STDOUT: - filename: prefix_repeat.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'Name', text: 'F'},
// CHECK:STDOUT:           {kind: 'ParameterListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ParameterList', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:                 {kind: 'NameExpression', text: 'n'},
// CHECK:STDOUT:               {kind: 'PrefixOperator', text: '&', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'PrefixOperator', text: '*', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'PrefixOperator', text: '*', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'PrefixOperator', text: '*', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ExpressionStatement', text: ';', subtree_size: 6},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 12},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
