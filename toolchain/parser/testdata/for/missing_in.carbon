// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{carbon} dump parse-tree %s 2>&1 | %{FileCheck-strict} %s
// AUTOUPDATE: %{carbon} dump parse-tree %s

fn foo() {
  // CHECK:{{.*}}/toolchain/parser/testdata/for/missing_in.carbon:[[@LINE+24]]:19: Expected `in` after loop `var` declaration.
  // CHECK:[
  // CHECK:{node_index: 19, kind: 'FunctionDeclaration', text: 'fn', subtree_size: 20, children: [
  // CHECK:  {node_index: 0, kind: 'DeclaredName', text: 'foo'},
  // CHECK:  {node_index: 2, kind: 'ParameterList', text: '(', subtree_size: 2, children: [
  // CHECK:    {node_index: 1, kind: 'ParameterListEnd', text: ')'}]},
  // CHECK:  {node_index: 18, kind: 'CodeBlock', text: '{', subtree_size: 16, children: [
  // CHECK:    {node_index: 16, kind: 'ForStatement', text: 'for', subtree_size: 14, children: [
  // CHECK:      {node_index: 8, kind: 'ForHeader', text: '(', has_error: yes, subtree_size: 6, children: [
  // CHECK:        {node_index: 6, kind: 'VariableDeclaration', text: 'var', subtree_size: 4, children: [
  // CHECK:          {node_index: 5, kind: 'PatternBinding', text: ':', subtree_size: 3, children: [
  // CHECK:            {node_index: 3, kind: 'DeclaredName', text: 'x'},
  // CHECK:            {node_index: 4, kind: 'Literal', text: 'i32'}]}]},
  // CHECK:        {node_index: 7, kind: 'ForHeaderEnd', text: ')'}]},
  // CHECK:      {node_index: 15, kind: 'CodeBlock', text: '{', subtree_size: 7, children: [
  // CHECK:        {node_index: 13, kind: 'ExpressionStatement', text: ';', subtree_size: 5, children: [
  // CHECK:          {node_index: 12, kind: 'CallExpression', text: '(', subtree_size: 4, children: [
  // CHECK:            {node_index: 9, kind: 'NameReference', text: 'Print'},
  // CHECK:            {node_index: 10, kind: 'NameReference', text: 'x'},
  // CHECK:            {node_index: 11, kind: 'CallExpressionEnd', text: ')'}]}]},
  // CHECK:        {node_index: 14, kind: 'CodeBlockEnd', text: '}'}]}]},
  // CHECK:    {node_index: 17, kind: 'CodeBlockEnd', text: '}'}]}]},
  // CHECK:{node_index: 20, kind: 'FileEnd', text: ''},
  // CHECK:]
  for (var x: i32 y) {
    Print(x);
  }
}
