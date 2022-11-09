// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{carbon-run-parser}
// CHECK:STDOUT: [
// CHECK:STDOUT:     {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:     {kind: 'DeclaredName', text: 'F'},
// CHECK:STDOUT:         {kind: 'DeclaredName', text: 'p'},
// CHECK:STDOUT:           {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:         {kind: 'PostfixOperator', text: '*', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'PatternBinding', text: ':', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'ParameterListComma', text: ','},
// CHECK:STDOUT:         {kind: 'DeclaredName', text: 'n'},
// CHECK:STDOUT:         {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:       {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'ParameterListEnd', text: ')'},
// CHECK:STDOUT:     {kind: 'ParameterList', text: '(', subtree_size: 10},
// CHECK:STDOUT:   {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 13},
// CHECK:STDOUT:       {kind: 'DeclaredName', text: 'q'},
// CHECK:STDOUT:         {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:       {kind: 'PostfixOperator', text: '*', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'PatternBinding', text: ':', subtree_size: 4},
// CHECK:STDOUT:       {kind: 'NameReference', text: 'p'},
// CHECK:STDOUT:     {kind: 'VariableInitializer', text: '=', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'DeclarationEnd', text: ';'},
// CHECK:STDOUT:   {kind: 'VariableDeclaration', text: 'var', subtree_size: 8},
// CHECK:STDOUT:       {kind: 'DeclaredName', text: 't'},
// CHECK:STDOUT:       {kind: 'NameReference', text: 'Type'},
// CHECK:STDOUT:     {kind: 'PatternBinding', text: ':', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:       {kind: 'PostfixOperator', text: '*', subtree_size: 2},
// CHECK:STDOUT:     {kind: 'VariableInitializer', text: '=', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'DeclarationEnd', text: ';'},
// CHECK:STDOUT:   {kind: 'VariableDeclaration', text: 'var', subtree_size: 8},
// CHECK:STDOUT:       {kind: 'NameReference', text: 't'},
// CHECK:STDOUT:           {kind: 'NameReference', text: 't'},
// CHECK:STDOUT:         {kind: 'PostfixOperator', text: '*', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'PostfixOperator', text: '*', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'InfixOperator', text: '=', subtree_size: 5},
// CHECK:STDOUT:   {kind: 'ExpressionStatement', text: ';', subtree_size: 6},
// CHECK:STDOUT:       {kind: 'NameReference', text: 'n'},
// CHECK:STDOUT:         {kind: 'NameReference', text: 'n'},
// CHECK:STDOUT:         {kind: 'NameReference', text: 'n'},
// CHECK:STDOUT:       {kind: 'InfixOperator', text: '*', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'InfixOperator', text: '=', subtree_size: 5},
// CHECK:STDOUT:   {kind: 'ExpressionStatement', text: ';', subtree_size: 6},
// CHECK:STDOUT:       {kind: 'NameReference', text: 'n'},
// CHECK:STDOUT:         {kind: 'NameReference', text: 'n'},
// CHECK:STDOUT:           {kind: 'NameReference', text: 'p'},
// CHECK:STDOUT:         {kind: 'PrefixOperator', text: '*', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'InfixOperator', text: '*', subtree_size: 4},
// CHECK:STDOUT:     {kind: 'InfixOperator', text: '=', subtree_size: 6},
// CHECK:STDOUT:   {kind: 'ExpressionStatement', text: ';', subtree_size: 7},
// CHECK:STDOUT:       {kind: 'NameReference', text: 'n'},
// CHECK:STDOUT:         {kind: 'NameReference', text: 'n'},
// CHECK:STDOUT:         {kind: 'NameReference', text: 'n'},
// CHECK:STDOUT:       {kind: 'InfixOperator', text: '*', subtree_size: 3},
// CHECK:STDOUT:     {kind: 'InfixOperator', text: '=', subtree_size: 5},
// CHECK:STDOUT:   {kind: 'ExpressionStatement', text: ';', subtree_size: 6},
// CHECK:STDOUT:       {kind: 'NameReference', text: 'G'},
// CHECK:STDOUT:         {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:       {kind: 'PostfixOperator', text: '*', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'CallExpressionComma', text: ','},
// CHECK:STDOUT:         {kind: 'NameReference', text: 'n'},
// CHECK:STDOUT:         {kind: 'NameReference', text: 'n'},
// CHECK:STDOUT:       {kind: 'InfixOperator', text: '*', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'CallExpressionEnd', text: ')'},
// CHECK:STDOUT:     {kind: 'CallExpression', text: '(', subtree_size: 9},
// CHECK:STDOUT:   {kind: 'ExpressionStatement', text: ';', subtree_size: 10},
// CHECK:STDOUT: {kind: 'FunctionDefinition', text: '}', subtree_size: 65},
// CHECK:STDOUT: {kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

fn F(p: i32*, n: i32) {
  var q: i32* = p;
  var t: Type = i32*;
  t = t**;
  n = n * n;
  n = n * *p;
  n = n*n;
  G(i32*, n * n);
}
