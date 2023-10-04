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
// CHECK:STDOUT:         {kind: 'Name', text: 'A'},
// CHECK:STDOUT:           {kind: 'ParameterListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ParameterList', text: ')', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:             {kind: 'PrefixOperator', text: 'const', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'PostfixOperator', text: '*', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'ReturnType', text: '->', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 9},
// CHECK:STDOUT:         {kind: 'ReturnStatementStart', text: 'return'},
// CHECK:STDOUT:             {kind: 'NameExpression', text: 'A'},
// CHECK:STDOUT:           {kind: 'CallExpressionStart', text: '(', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'CallExpression', text: ')', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ReturnStatement', text: ';', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 15},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'Name', text: 'B'},
// CHECK:STDOUT:           {kind: 'ParameterListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ParameterList', text: ')', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'ParenExpressionOrTupleLiteralStart', text: '('},
// CHECK:STDOUT:                 {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:               {kind: 'PrefixOperator', text: 'const', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'ParenExpression', text: ')', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'PostfixOperator', text: '*', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'ReturnType', text: '->', subtree_size: 6},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 11},
// CHECK:STDOUT:         {kind: 'ReturnStatementStart', text: 'return'},
// CHECK:STDOUT:             {kind: 'NameExpression', text: 'B'},
// CHECK:STDOUT:           {kind: 'CallExpressionStart', text: '(', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'CallExpression', text: ')', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ReturnStatement', text: ';', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 17},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'Name', text: 'C'},
// CHECK:STDOUT:           {kind: 'ParameterListStart', text: '('},
// CHECK:STDOUT:         {kind: 'ParameterList', text: ')', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'ParenExpressionOrTupleLiteralStart', text: '('},
// CHECK:STDOUT:                 {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:               {kind: 'PostfixOperator', text: '*', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'ParenExpression', text: ')', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'PrefixOperator', text: 'const', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'ReturnType', text: '->', subtree_size: 6},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 11},
// CHECK:STDOUT:         {kind: 'ReturnStatementStart', text: 'return'},
// CHECK:STDOUT:             {kind: 'NameExpression', text: 'C'},
// CHECK:STDOUT:           {kind: 'CallExpressionStart', text: '(', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'CallExpression', text: ')', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ReturnStatement', text: ';', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 17},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
