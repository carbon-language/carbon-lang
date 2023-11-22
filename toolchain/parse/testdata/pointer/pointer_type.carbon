// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F(p: i32*) -> i32* {
  return p;
}

var T: type = if true then i32* else f64*;

// CHECK:STDOUT: - filename: pointer_type.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'Name', text: 'F'},
// CHECK:STDOUT:           {kind: 'ParamListStart', text: '('},
// CHECK:STDOUT:             {kind: 'Name', text: 'p'},
// CHECK:STDOUT:               {kind: 'IntegerTypeLiteral', text: 'i32'},
// CHECK:STDOUT:             {kind: 'PostfixOperator', text: '*', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'PatternBinding', text: ':', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'ParamList', text: ')', subtree_size: 6},
// CHECK:STDOUT:             {kind: 'IntegerTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'PostfixOperator', text: '*', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'ReturnType', text: '->', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 12},
// CHECK:STDOUT:         {kind: 'ReturnStatementStart', text: 'return'},
// CHECK:STDOUT:         {kind: 'NameExpr', text: 'p'},
// CHECK:STDOUT:       {kind: 'ReturnStatement', text: ';', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 16},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:         {kind: 'Name', text: 'T'},
// CHECK:STDOUT:         {kind: 'TypeTypeLiteral', text: 'type'},
// CHECK:STDOUT:       {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'VariableInitializer', text: '='},
// CHECK:STDOUT:           {kind: 'BoolLiteralTrue', text: 'true'},
// CHECK:STDOUT:         {kind: 'IfExprIf', text: 'if', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'IntegerTypeLiteral', text: 'i32'},
// CHECK:STDOUT:           {kind: 'PostfixOperator', text: '*', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'IfExprThen', text: 'then', subtree_size: 3},
// CHECK:STDOUT:           {kind: 'FloatingPointTypeLiteral', text: 'f64'},
// CHECK:STDOUT:         {kind: 'PostfixOperator', text: '*', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'IfExprElse', text: 'else', subtree_size: 8},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: ';', subtree_size: 14},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
