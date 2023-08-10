// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F(c: bool) {
  var a: i32;
  var b: i32;
  var p: i32*;
  *p = if c then 1 else 2;
  // These are valid to _parse_ even though rejected semantically.
  (if c then a else b) += if c then 1 else 2;
  ++if c then a else b;
}

// CHECK:STDOUT: [
// CHECK:STDOUT:     {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:     {kind: 'Name', text: 'F'},
// CHECK:STDOUT:       {kind: 'ParameterListStart', text: '('},
// CHECK:STDOUT:         {kind: 'Name', text: 'c'},
// CHECK:STDOUT:         {kind: 'Literal', text: 'bool'},
// CHECK:STDOUT:       {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'ParameterList', text: ')', subtree_size: 5},
// CHECK:STDOUT:   {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 8},
// CHECK:STDOUT:     {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:       {kind: 'Name', text: 'a'},
// CHECK:STDOUT:       {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:     {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:   {kind: 'VariableDeclaration', text: ';', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:       {kind: 'Name', text: 'b'},
// CHECK:STDOUT:       {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:     {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:   {kind: 'VariableDeclaration', text: ';', subtree_size: 5},
// CHECK:STDOUT:     {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:       {kind: 'Name', text: 'p'},
// CHECK:STDOUT:         {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:       {kind: 'PostfixOperator', text: '*', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'PatternBinding', text: ':', subtree_size: 4},
// CHECK:STDOUT:   {kind: 'VariableDeclaration', text: ';', subtree_size: 6},
// CHECK:STDOUT:         {kind: 'NameExpression', text: 'p'},
// CHECK:STDOUT:       {kind: 'PrefixOperator', text: '*', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'NameExpression', text: 'c'},
// CHECK:STDOUT:         {kind: 'IfExpressionIf', text: 'if', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'Literal', text: '1'},
// CHECK:STDOUT:         {kind: 'IfExpressionThen', text: 'then', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'Literal', text: '2'},
// CHECK:STDOUT:       {kind: 'IfExpressionElse', text: 'else', subtree_size: 6},
// CHECK:STDOUT:     {kind: 'InfixOperator', text: '=', subtree_size: 9},
// CHECK:STDOUT:   {kind: 'ExpressionStatement', text: ';', subtree_size: 10},
// CHECK:STDOUT:         {kind: 'ParenExpressionOrTupleLiteralStart', text: '('},
// CHECK:STDOUT:             {kind: 'NameExpression', text: 'c'},
// CHECK:STDOUT:           {kind: 'IfExpressionIf', text: 'if', subtree_size: 2},
// CHECK:STDOUT:             {kind: 'NameExpression', text: 'a'},
// CHECK:STDOUT:           {kind: 'IfExpressionThen', text: 'then', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'NameExpression', text: 'b'},
// CHECK:STDOUT:         {kind: 'IfExpressionElse', text: 'else', subtree_size: 6},
// CHECK:STDOUT:       {kind: 'ParenExpression', text: ')', subtree_size: 8},
// CHECK:STDOUT:           {kind: 'NameExpression', text: 'c'},
// CHECK:STDOUT:         {kind: 'IfExpressionIf', text: 'if', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'Literal', text: '1'},
// CHECK:STDOUT:         {kind: 'IfExpressionThen', text: 'then', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'Literal', text: '2'},
// CHECK:STDOUT:       {kind: 'IfExpressionElse', text: 'else', subtree_size: 6},
// CHECK:STDOUT:     {kind: 'InfixOperator', text: '+=', subtree_size: 15},
// CHECK:STDOUT:   {kind: 'ExpressionStatement', text: ';', subtree_size: 16},
// CHECK:STDOUT:           {kind: 'NameExpression', text: 'c'},
// CHECK:STDOUT:         {kind: 'IfExpressionIf', text: 'if', subtree_size: 2},
// CHECK:STDOUT:           {kind: 'NameExpression', text: 'a'},
// CHECK:STDOUT:         {kind: 'IfExpressionThen', text: 'then', subtree_size: 2},
// CHECK:STDOUT:         {kind: 'NameExpression', text: 'b'},
// CHECK:STDOUT:       {kind: 'IfExpressionElse', text: 'else', subtree_size: 6},
// CHECK:STDOUT:     {kind: 'PrefixOperator', text: '++', subtree_size: 7},
// CHECK:STDOUT:   {kind: 'ExpressionStatement', text: ';', subtree_size: 8},
// CHECK:STDOUT: {kind: 'FunctionDefinition', text: '}', subtree_size: 59},
// CHECK:STDOUT: {kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]
