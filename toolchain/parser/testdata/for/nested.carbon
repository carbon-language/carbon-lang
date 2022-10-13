// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{carbon} dump parse-tree %s | %{FileCheck-strict} %s
// AUTOUPDATE: %{carbon} dump parse-tree %s
// CHECK:[
// CHECK:{node_index: 32, kind: 'FunctionDeclaration', text: 'fn', subtree_size: 33, children: [
// CHECK:  {node_index: 0, kind: 'DeclaredName', text: 'foo'},
// CHECK:  {node_index: 2, kind: 'ParameterList', text: '(', subtree_size: 2, children: [
// CHECK:    {node_index: 1, kind: 'ParameterListEnd', text: ')'}]},
// CHECK:  {node_index: 31, kind: 'CodeBlock', text: '{', subtree_size: 29, children: [
// CHECK:    {node_index: 29, kind: 'ForStatement', text: 'for', subtree_size: 27, children: [
// CHECK:      {node_index: 10, kind: 'ForHeader', text: '(', subtree_size: 8, children: [
// CHECK:        {node_index: 6, kind: 'VariableDeclaration', text: 'var', subtree_size: 4, children: [
// CHECK:          {node_index: 5, kind: 'PatternBinding', text: ':', subtree_size: 3, children: [
// CHECK:            {node_index: 3, kind: 'DeclaredName', text: 'y'},
// CHECK:            {node_index: 4, kind: 'Literal', text: 'i32'}]}]},
// CHECK:        {node_index: 7, kind: 'ForIn', text: 'in'},
// CHECK:        {node_index: 8, kind: 'NameReference', text: 'x'},
// CHECK:        {node_index: 9, kind: 'ForHeaderEnd', text: ')'}]},
// CHECK:      {node_index: 28, kind: 'CodeBlock', text: '{', subtree_size: 18, children: [
// CHECK:        {node_index: 26, kind: 'ForStatement', text: 'for', subtree_size: 16, children: [
// CHECK:          {node_index: 18, kind: 'ForHeader', text: '(', subtree_size: 8, children: [
// CHECK:            {node_index: 14, kind: 'VariableDeclaration', text: 'var', subtree_size: 4, children: [
// CHECK:              {node_index: 13, kind: 'PatternBinding', text: ':', subtree_size: 3, children: [
// CHECK:                {node_index: 11, kind: 'DeclaredName', text: 'z'},
// CHECK:                {node_index: 12, kind: 'Literal', text: 'i32'}]}]},
// CHECK:            {node_index: 15, kind: 'ForIn', text: 'in'},
// CHECK:            {node_index: 16, kind: 'NameReference', text: 'y'},
// CHECK:            {node_index: 17, kind: 'ForHeaderEnd', text: ')'}]},
// CHECK:          {node_index: 25, kind: 'CodeBlock', text: '{', subtree_size: 7, children: [
// CHECK:            {node_index: 23, kind: 'ExpressionStatement', text: ';', subtree_size: 5, children: [
// CHECK:              {node_index: 22, kind: 'CallExpression', text: '(', subtree_size: 4, children: [
// CHECK:                {node_index: 19, kind: 'NameReference', text: 'Print'},
// CHECK:                {node_index: 20, kind: 'NameReference', text: 'z'},
// CHECK:                {node_index: 21, kind: 'CallExpressionEnd', text: ')'}]}]},
// CHECK:            {node_index: 24, kind: 'CodeBlockEnd', text: '}'}]}]},
// CHECK:        {node_index: 27, kind: 'CodeBlockEnd', text: '}'}]}]},
// CHECK:    {node_index: 30, kind: 'CodeBlockEnd', text: '}'}]}]},
// CHECK:{node_index: 33, kind: 'FileEnd', text: ''},
// CHECK:]

fn foo() {
  for (var y: i32 in x) {
      for (var z: i32 in y) {
          Print(z);
      }
  }
}
