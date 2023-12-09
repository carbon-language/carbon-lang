// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() {
  i32****;
}

// CHECK:STDOUT: - filename: postfix_repeat.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'F'},
// CHECK:STDOUT:           {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:         {kind: 'TuplePattern', text: ')', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:                 {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:               {kind: 'PostfixOperator', text: '*', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'PostfixOperator', text: '*', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'PostfixOperator', text: '*', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'PostfixOperator', text: '*', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'ExprStatement', text: ';', subtree_size: 6},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 12},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
