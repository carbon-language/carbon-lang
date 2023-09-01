// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var n: i32;

fn F() -> if true then i32 else i32* {
  return if true then n else &n;
}

// CHECK:STDOUT: [
// CHECK:STDOUT:   {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:     {kind: 'Name', text: 'n'},
// CHECK:STDOUT:     {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:   {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT: {kind: 'VariableDeclaration', text: ';', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:     {kind: 'Name', text: 'F'},
// CHECK:STDOUT:       {kind: 'ParameterListStart', text: '('},
// CHECK:STDOUT:     {kind: 'ParameterList', text: ')', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'Literal', text: 'true'},
// CHECK:STDOUT:         {kind: 'IfExpressionIf', text: 'if', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:         {kind: 'IfExpressionThen', text: 'then', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:         {kind: 'PostfixOperator', text: '*', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'IfExpressionElse', text: 'else', subtree_size: 7},
// CHECK:STDOUT:     {kind: 'ReturnType', text: '->', subtree_size: 8},
// CHECK:STDOUT:   {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 13},
// CHECK:STDOUT:     {kind: 'ReturnStatementStart', text: 'return'},
// CHECK:STDOUT:         {kind: 'Literal', text: 'true'},
// CHECK:STDOUT:       {kind: 'IfExpressionIf', text: 'if', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'NameExpression', text: 'n'},
// CHECK:STDOUT:       {kind: 'IfExpressionThen', text: 'then', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'NameExpression', text: 'n'},
// CHECK:STDOUT:       {kind: 'PrefixOperator', text: '&', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'IfExpressionElse', text: 'else', subtree_size: 7},
// CHECK:STDOUT:   {kind: 'ReturnStatement', text: ';', subtree_size: 9},
// CHECK:STDOUT: {kind: 'FunctionDefinition', text: '}', subtree_size: 23},
// CHECK:STDOUT: {kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]
