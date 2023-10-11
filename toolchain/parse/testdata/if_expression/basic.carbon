// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F(b: bool, x: i32, y: i32) -> bool {
  return if b then x else y;
}

// CHECK:STDOUT: - filename: basic.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'Name', text: 'F'},
// CHECK:STDOUT:           {kind: 'ParameterListStart', text: '('},
// CHECK:STDOUT:             {kind: 'Name', text: 'b'},
// CHECK:STDOUT:             {kind: 'Literal', text: 'bool'},
// CHECK:STDOUT:           {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'ParameterListComma', text: ','},
// CHECK:STDOUT:             {kind: 'Name', text: 'x'},
// CHECK:STDOUT:             {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:           {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'ParameterListComma', text: ','},
// CHECK:STDOUT:             {kind: 'Name', text: 'y'},
// CHECK:STDOUT:             {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:           {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'ParameterList', text: ')', subtree_size: 13},
// CHECK:STDOUT:           {kind: 'Literal', text: 'bool'},
// CHECK:STDOUT:         {kind: 'ReturnType', text: '->', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 18},
// CHECK:STDOUT:         {kind: 'ReturnStatementStart', text: 'return'},
// CHECK:STDOUT:             {kind: 'NameExpression', text: 'b'},
// CHECK:STDOUT:           {kind: 'IfExpressionIf', text: 'if', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'NameExpression', text: 'x'},
// CHECK:STDOUT:           {kind: 'IfExpressionThen', text: 'then', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'NameExpression', text: 'y'},
// CHECK:STDOUT:         {kind: 'IfExpressionElse', text: 'else', subtree_size: 6},
// CHECK:STDOUT:       {kind: 'ReturnStatement', text: ';', subtree_size: 8},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 27},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
