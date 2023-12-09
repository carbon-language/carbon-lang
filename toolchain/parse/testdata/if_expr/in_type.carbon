// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var n: i32;

fn F() -> if true then i32 else i32* {
  return if true then n else &n;
}

// CHECK:STDOUT: - filename: in_type.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'n'},
// CHECK:STDOUT:         {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:       {kind: 'BindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'VariableDecl', text: ';', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'F'},
// CHECK:STDOUT:           {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:         {kind: 'TuplePattern', text: ')', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'BoolLiteralTrue', text: 'true'},
// CHECK:STDOUT:             {kind: 'IfExprIf', text: 'if', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:             {kind: 'IfExprThen', text: 'then', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:             {kind: 'PostfixOperator', text: '*', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'IfExprElse', text: 'else', subtree_size: 7},
// CHECK:STDOUT:         {kind: 'ReturnType', text: '->', subtree_size: 8},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 13},
// CHECK:STDOUT:         {kind: 'ReturnStatementStart', text: 'return'},
// CHECK:STDOUT:             {kind: 'BoolLiteralTrue', text: 'true'},
// CHECK:STDOUT:           {kind: 'IfExprIf', text: 'if', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'n'},
// CHECK:STDOUT:           {kind: 'IfExprThen', text: 'then', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'n'},
// CHECK:STDOUT:           {kind: 'PrefixOperator', text: '&', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'IfExprElse', text: 'else', subtree_size: 7},
// CHECK:STDOUT:       {kind: 'ReturnStatement', text: ';', subtree_size: 9},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 23},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
