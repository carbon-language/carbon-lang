// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn A() -> const i32* { return A(); }
fn B() -> (const i32)* { return B(); }
fn C() -> const (i32*) { return C(); }

// CHECK:STDOUT: - filename: const_pointer.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'A'},
// CHECK:STDOUT:           {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:         {kind: 'TuplePattern', text: ')', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:             {kind: 'PrefixOperator', text: 'const', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'PostfixOperator', text: '*', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'ReturnType', text: '->', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 9},
// CHECK:STDOUT:         {kind: 'ReturnStatementStart', text: 'return'},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'A'},
// CHECK:STDOUT:           {kind: 'CallExprStart', text: '(', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'CallExpr', text: ')', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ReturnStatement', text: ';', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 15},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'B'},
// CHECK:STDOUT:           {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:         {kind: 'TuplePattern', text: ')', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'ExprOpenParen', text: '('},
// CHECK:STDOUT:                 {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:               {kind: 'PrefixOperator', text: 'const', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'ParenExpr', text: ')', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'PostfixOperator', text: '*', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'ReturnType', text: '->', subtree_size: 6},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 11},
// CHECK:STDOUT:         {kind: 'ReturnStatementStart', text: 'return'},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'B'},
// CHECK:STDOUT:           {kind: 'CallExprStart', text: '(', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'CallExpr', text: ')', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ReturnStatement', text: ';', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 17},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'C'},
// CHECK:STDOUT:           {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:         {kind: 'TuplePattern', text: ')', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'ExprOpenParen', text: '('},
// CHECK:STDOUT:                 {kind: 'IntTypeLiteral', text: 'i32'},
// CHECK:STDOUT:               {kind: 'PostfixOperator', text: '*', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'ParenExpr', text: ')', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'PrefixOperator', text: 'const', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'ReturnType', text: '->', subtree_size: 6},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 11},
// CHECK:STDOUT:         {kind: 'ReturnStatementStart', text: 'return'},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'C'},
// CHECK:STDOUT:           {kind: 'CallExprStart', text: '(', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'CallExpr', text: ')', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ReturnStatement', text: ';', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 17},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
