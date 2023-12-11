// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F(b: bool) -> bool {
  return if b and b then b and b else b or b;
}

// CHECK:STDOUT: - filename: precedence.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'F'},
// CHECK:STDOUT:           {kind: 'TuplePatternStart', text: '('},
// CHECK:STDOUT:             {kind: 'IdentifierName', text: 'b'},
// CHECK:STDOUT:             {kind: 'BoolTypeLiteral', text: 'bool'},
// CHECK:STDOUT:           {kind: 'BindingPattern', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'TuplePattern', text: ')', subtree_size: 5},
// CHECK:STDOUT:           {kind: 'BoolTypeLiteral', text: 'bool'},
// CHECK:STDOUT:         {kind: 'ReturnType', text: '->', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 10},
// CHECK:STDOUT:         {kind: 'ReturnStatementStart', text: 'return'},
// CHECK:STDOUT:                 {kind: 'IdentifierNameExpr', text: 'b'},
// CHECK:STDOUT:               {kind: 'ShortCircuitOperandAnd', text: 'and', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'IdentifierNameExpr', text: 'b'},
// CHECK:STDOUT:             {kind: 'ShortCircuitOperatorAnd', text: 'and', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'IfExprIf', text: 'if', subtree_size: 5},
// CHECK:STDOUT:                 {kind: 'IdentifierNameExpr', text: 'b'},
// CHECK:STDOUT:               {kind: 'ShortCircuitOperandAnd', text: 'and', subtree_size: 2},
// CHECK:STDOUT:               {kind: 'IdentifierNameExpr', text: 'b'},
// CHECK:STDOUT:             {kind: 'ShortCircuitOperatorAnd', text: 'and', subtree_size: 4},
// CHECK:STDOUT:           {kind: 'IfExprThen', text: 'then', subtree_size: 5},
// CHECK:STDOUT:               {kind: 'IdentifierNameExpr', text: 'b'},
// CHECK:STDOUT:             {kind: 'ShortCircuitOperandOr', text: 'or', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'IdentifierNameExpr', text: 'b'},
// CHECK:STDOUT:           {kind: 'ShortCircuitOperatorOr', text: 'or', subtree_size: 4},
// CHECK:STDOUT:         {kind: 'IfExprElse', text: 'else', subtree_size: 15},
// CHECK:STDOUT:       {kind: 'ReturnStatement', text: ';', subtree_size: 17},
// CHECK:STDOUT:     {kind: 'FunctionDefinition', text: '}', subtree_size: 28},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
