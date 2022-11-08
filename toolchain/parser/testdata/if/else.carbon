// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{carbon-run-parser}
// CHECK:STDOUT: [
// CHECK:STDOUT:     {kind : 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:     {kind : 'DeclaredName', text: 'F'},
// CHECK:STDOUT:       {kind : 'ParameterListEnd', text: ')'},
// CHECK:STDOUT:     {kind : 'ParameterList', text: '(', subtree_size: 2},
// CHECK:STDOUT:   {kind : 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:       {kind : 'NameReference', text: 'a'},
// CHECK:STDOUT:       {kind : 'ConditionEnd', text: ')'},
// CHECK:STDOUT:     {kind : 'Condition', text: '(', subtree_size: 3},
// CHECK:STDOUT:       {kind : 'CodeBlockStart', text: '{'},
// CHECK:STDOUT:           {kind : 'NameReference', text: 'b'},
// CHECK:STDOUT:           {kind : 'ConditionEnd', text: ')'},
// CHECK:STDOUT:         {kind : 'Condition', text: '(', subtree_size: 3},
// CHECK:STDOUT:           {kind : 'CodeBlockStart', text: '{'},
// CHECK:STDOUT:             {kind : 'NameReference', text: 'c'},
// CHECK:STDOUT:           {kind : 'ExpressionStatement', text: ';', subtree_size: 2},
// CHECK:STDOUT:         {kind : 'CodeBlock', text: '}', subtree_size: 4},
// CHECK:STDOUT:         {kind : 'IfStatementElse', text: 'else'},
// CHECK:STDOUT:           {kind : 'CodeBlockStart', text: '{'},
// CHECK:STDOUT:             {kind : 'NameReference', text: 'd'},
// CHECK:STDOUT:           {kind : 'ExpressionStatement', text: ';', subtree_size: 2},
// CHECK:STDOUT:         {kind : 'CodeBlock', text: '}', subtree_size: 4},
// CHECK:STDOUT:       {kind : 'IfStatement', text: 'if', subtree_size: 13},
// CHECK:STDOUT:     {kind : 'CodeBlock', text: '}', subtree_size: 15},
// CHECK:STDOUT:     {kind : 'IfStatementElse', text: 'else'},
// CHECK:STDOUT:       {kind : 'CodeBlockStart', text: '{'},
// CHECK:STDOUT:         {kind : 'NameReference', text: 'e'},
// CHECK:STDOUT:       {kind : 'ExpressionStatement', text: ';', subtree_size: 2},
// CHECK:STDOUT:     {kind : 'CodeBlock', text: '}', subtree_size: 4},
// CHECK:STDOUT:   {kind : 'IfStatement', text: 'if', subtree_size: 24},
// CHECK:STDOUT:       {kind : 'NameReference', text: 'x'},
// CHECK:STDOUT:       {kind : 'ConditionEnd', text: ')'},
// CHECK:STDOUT:     {kind : 'Condition', text: '(', subtree_size: 3},
// CHECK:STDOUT:       {kind : 'CodeBlockStart', text: '{'},
// CHECK:STDOUT:           {kind : 'NameReference', text: 'G'},
// CHECK:STDOUT:           {kind : 'Literal', text: '1'},
// CHECK:STDOUT:           {kind : 'CallExpressionEnd', text: ')'},
// CHECK:STDOUT:         {kind : 'CallExpression', text: '(', subtree_size: 4},
// CHECK:STDOUT:       {kind : 'ExpressionStatement', text: ';', subtree_size: 5},
// CHECK:STDOUT:     {kind : 'CodeBlock', text: '}', subtree_size: 7},
// CHECK:STDOUT:     {kind : 'IfStatementElse', text: 'else'},
// CHECK:STDOUT:         {kind : 'NameReference', text: 'x'},
// CHECK:STDOUT:         {kind : 'ConditionEnd', text: ')'},
// CHECK:STDOUT:       {kind : 'Condition', text: '(', subtree_size: 3},
// CHECK:STDOUT:         {kind : 'CodeBlockStart', text: '{'},
// CHECK:STDOUT:             {kind : 'NameReference', text: 'G'},
// CHECK:STDOUT:             {kind : 'Literal', text: '2'},
// CHECK:STDOUT:             {kind : 'CallExpressionEnd', text: ')'},
// CHECK:STDOUT:           {kind : 'CallExpression', text: '(', subtree_size: 4},
// CHECK:STDOUT:         {kind : 'ExpressionStatement', text: ';', subtree_size: 5},
// CHECK:STDOUT:       {kind : 'CodeBlock', text: '}', subtree_size: 7},
// CHECK:STDOUT:       {kind : 'IfStatementElse', text: 'else'},
// CHECK:STDOUT:         {kind : 'CodeBlockStart', text: '{'},
// CHECK:STDOUT:             {kind : 'NameReference', text: 'G'},
// CHECK:STDOUT:             {kind : 'Literal', text: '3'},
// CHECK:STDOUT:             {kind : 'CallExpressionEnd', text: ')'},
// CHECK:STDOUT:           {kind : 'CallExpression', text: '(', subtree_size: 4},
// CHECK:STDOUT:         {kind : 'ExpressionStatement', text: ';', subtree_size: 5},
// CHECK:STDOUT:       {kind : 'CodeBlock', text: '}', subtree_size: 7},
// CHECK:STDOUT:     {kind : 'IfStatement', text: 'if', subtree_size: 19},
// CHECK:STDOUT:   {kind : 'IfStatement', text: 'if', subtree_size: 31},
// CHECK:STDOUT: {kind : 'FunctionDefinition', text: '}', subtree_size: 61},
// CHECK:STDOUT: {kind : 'FileEnd', text: ''},
// CHECK:STDOUT: ]

fn F() {
  if (a) {
    if (b) {
      c;
    } else {
      d;
    }
  } else {
    e;
  }
  if (x) { G(1); }
  else if (x) { G(2); }
  else { G(3); }
}
