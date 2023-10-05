// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> i32 {
  var n: i32 = 0;
  var p: i32* = &n;
  var q: i32** = &p;
  return **q;
}

// CHECK:STDOUT: - filename: pointer_value.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'Name', text: 'F'},
// CHECK:STDOUT:           {kind: 'ParameterListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ParameterList', text: ')', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:         {kind: 'ReturnType', text: '->', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 7},
// CHECK:STDOUT:         {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:           {kind: 'Name', text: 'n'},
// CHECK:STDOUT:           {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:         {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:         {kind: 'Literal', text: '0'},
// CHECK:STDOUT:       {kind: 'VariableDeclaration', text: ';', subtree_size: 7},
// CHECK:STDOUT:         {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:           {kind: 'Name', text: 'p'},
// CHECK:STDOUT:             {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:           {kind: 'PostfixOperator', text: '*', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'PatternBinding', text: ':', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:           {kind: 'NameExpression', text: 'n'},
// CHECK:STDOUT:         {kind: 'PrefixOperator', text: '&', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'VariableDeclaration', text: ';', subtree_size: 9},
// CHECK:STDOUT:         {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:           {kind: 'Name', text: 'q'},
// CHECK:STDOUT:               {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:             {kind: 'PostfixOperator', text: '*', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'PostfixOperator', text: '*', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'PatternBinding', text: ':', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:           {kind: 'NameExpression', text: 'p'},
// CHECK:STDOUT:         {kind: 'PrefixOperator', text: '&', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'VariableDeclaration', text: ';', subtree_size: 10},
// CHECK:STDOUT:         {kind: 'ReturnStatementStart', text: 'return'},
// CHECK:STDOUT:             {kind: 'NameExpression', text: 'q'},
// CHECK:STDOUT:           {kind: 'PrefixOperator', text: '*', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'PrefixOperator', text: '*', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ReturnStatement', text: ';', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 39},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
