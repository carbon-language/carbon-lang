// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{carbon-run-parser}
// CHECK:STDOUT: [
// CHECK:STDOUT: {node_index: 60, kind: 'FunctionDefinition', text: '}', subtree_size: 61, children: [
// CHECK:STDOUT:   {node_index: 4, kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5, children: [
// CHECK:STDOUT:     {node_index: 0, kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:     {node_index: 1, kind: 'DeclaredName', text: 'F'},
// CHECK:STDOUT:     {node_index: 3, kind: 'ParameterList', text: '(', subtree_size: 2, children: [
// CHECK:STDOUT:       {node_index: 2, kind: 'ParameterListEnd', text: ')'}]}]},
// CHECK:STDOUT:   {node_index: 28, kind: 'IfStatement', text: 'if', subtree_size: 24, children: [
// CHECK:STDOUT:     {node_index: 7, kind: 'Condition', text: '(', subtree_size: 3, children: [
// CHECK:STDOUT:       {node_index: 5, kind: 'NameReference', text: 'a'},
// CHECK:STDOUT:       {node_index: 6, kind: 'ConditionEnd', text: ')'}]},
// CHECK:STDOUT:     {node_index: 22, kind: 'CodeBlock', text: '}', subtree_size: 15, children: [
// CHECK:STDOUT:       {node_index: 8, kind: 'CodeBlockStart', text: '{'},
// CHECK:STDOUT:       {node_index: 21, kind: 'IfStatement', text: 'if', subtree_size: 13, children: [
// CHECK:STDOUT:         {node_index: 11, kind: 'Condition', text: '(', subtree_size: 3, children: [
// CHECK:STDOUT:           {node_index: 9, kind: 'NameReference', text: 'b'},
// CHECK:STDOUT:           {node_index: 10, kind: 'ConditionEnd', text: ')'}]},
// CHECK:STDOUT:         {node_index: 15, kind: 'CodeBlock', text: '}', subtree_size: 4, children: [
// CHECK:STDOUT:           {node_index: 12, kind: 'CodeBlockStart', text: '{'},
// CHECK:STDOUT:           {node_index: 14, kind: 'ExpressionStatement', text: ';', subtree_size: 2, children: [
// CHECK:STDOUT:             {node_index: 13, kind: 'NameReference', text: 'c'}]}]},
// CHECK:STDOUT:         {node_index: 16, kind: 'IfStatementElse', text: 'else'},
// CHECK:STDOUT:         {node_index: 20, kind: 'CodeBlock', text: '}', subtree_size: 4, children: [
// CHECK:STDOUT:           {node_index: 17, kind: 'CodeBlockStart', text: '{'},
// CHECK:STDOUT:           {node_index: 19, kind: 'ExpressionStatement', text: ';', subtree_size: 2, children: [
// CHECK:STDOUT:             {node_index: 18, kind: 'NameReference', text: 'd'}]}]}]}]},
// CHECK:STDOUT:     {node_index: 23, kind: 'IfStatementElse', text: 'else'},
// CHECK:STDOUT:     {node_index: 27, kind: 'CodeBlock', text: '}', subtree_size: 4, children: [
// CHECK:STDOUT:       {node_index: 24, kind: 'CodeBlockStart', text: '{'},
// CHECK:STDOUT:       {node_index: 26, kind: 'ExpressionStatement', text: ';', subtree_size: 2, children: [
// CHECK:STDOUT:         {node_index: 25, kind: 'NameReference', text: 'e'}]}]}]},
// CHECK:STDOUT:   {node_index: 59, kind: 'IfStatement', text: 'if', subtree_size: 31, children: [
// CHECK:STDOUT:     {node_index: 31, kind: 'Condition', text: '(', subtree_size: 3, children: [
// CHECK:STDOUT:       {node_index: 29, kind: 'NameReference', text: 'x'},
// CHECK:STDOUT:       {node_index: 30, kind: 'ConditionEnd', text: ')'}]},
// CHECK:STDOUT:     {node_index: 38, kind: 'CodeBlock', text: '}', subtree_size: 7, children: [
// CHECK:STDOUT:       {node_index: 32, kind: 'CodeBlockStart', text: '{'},
// CHECK:STDOUT:       {node_index: 37, kind: 'ExpressionStatement', text: ';', subtree_size: 5, children: [
// CHECK:STDOUT:         {node_index: 36, kind: 'CallExpression', text: '(', subtree_size: 4, children: [
// CHECK:STDOUT:           {node_index: 33, kind: 'NameReference', text: 'G'},
// CHECK:STDOUT:           {node_index: 34, kind: 'Literal', text: '1'},
// CHECK:STDOUT:           {node_index: 35, kind: 'CallExpressionEnd', text: ')'}]}]}]},
// CHECK:STDOUT:     {node_index: 39, kind: 'IfStatementElse', text: 'else'},
// CHECK:STDOUT:     {node_index: 58, kind: 'IfStatement', text: 'if', subtree_size: 19, children: [
// CHECK:STDOUT:       {node_index: 42, kind: 'Condition', text: '(', subtree_size: 3, children: [
// CHECK:STDOUT:         {node_index: 40, kind: 'NameReference', text: 'x'},
// CHECK:STDOUT:         {node_index: 41, kind: 'ConditionEnd', text: ')'}]},
// CHECK:STDOUT:       {node_index: 49, kind: 'CodeBlock', text: '}', subtree_size: 7, children: [
// CHECK:STDOUT:         {node_index: 43, kind: 'CodeBlockStart', text: '{'},
// CHECK:STDOUT:         {node_index: 48, kind: 'ExpressionStatement', text: ';', subtree_size: 5, children: [
// CHECK:STDOUT:           {node_index: 47, kind: 'CallExpression', text: '(', subtree_size: 4, children: [
// CHECK:STDOUT:             {node_index: 44, kind: 'NameReference', text: 'G'},
// CHECK:STDOUT:             {node_index: 45, kind: 'Literal', text: '2'},
// CHECK:STDOUT:             {node_index: 46, kind: 'CallExpressionEnd', text: ')'}]}]}]},
// CHECK:STDOUT:       {node_index: 50, kind: 'IfStatementElse', text: 'else'},
// CHECK:STDOUT:       {node_index: 57, kind: 'CodeBlock', text: '}', subtree_size: 7, children: [
// CHECK:STDOUT:         {node_index: 51, kind: 'CodeBlockStart', text: '{'},
// CHECK:STDOUT:         {node_index: 56, kind: 'ExpressionStatement', text: ';', subtree_size: 5, children: [
// CHECK:STDOUT:           {node_index: 55, kind: 'CallExpression', text: '(', subtree_size: 4, children: [
// CHECK:STDOUT:             {node_index: 52, kind: 'NameReference', text: 'G'},
// CHECK:STDOUT:             {node_index: 53, kind: 'Literal', text: '3'},
// CHECK:STDOUT:             {node_index: 54, kind: 'CallExpressionEnd', text: ')'}]}]}]}]}]}]},
// CHECK:STDOUT: {node_index: 61, kind: 'FileEnd', text: ''},
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
