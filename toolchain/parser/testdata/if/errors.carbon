// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{carbon} dump parse-tree %s | %{FileCheck-strict} %s
// AUTOUPDATE: %{carbon} dump parse-tree %s

fn F() {
  // CHECK:{{.*}}/toolchain/parser/testdata/if/errors.carbon:[[@LINE+1]]:6: Expected `(` after `if`.
  if a {}
  // CHECK:{{.*}}/toolchain/parser/testdata/if/errors.carbon:[[@LINE+1]]:7: Expected expression.
  if () {}
  // CHECK:{{.*}}/toolchain/parser/testdata/if/errors.carbon:[[@LINE+1]]:9: Unexpected tokens before `)`.
  if (b c) {}
  if (d)
// CHECK:{{.*}}/toolchain/parser/testdata/if/errors.carbon:[[@LINE+30]]:1: Expected braced code block.
// CHECK:{{.*}}/toolchain/parser/testdata/if/errors.carbon:[[@LINE+29]]:1: Expected expression.
// CHECK:[
// CHECK:{node_index: 24, kind: 'FunctionDeclaration', text: 'fn', subtree_size: 25, children: [
// CHECK:  {node_index: 0, kind: 'DeclaredName', text: 'F'},
// CHECK:  {node_index: 2, kind: 'ParameterList', text: '(', subtree_size: 2, children: [
// CHECK:    {node_index: 1, kind: 'ParameterListEnd', text: ')'}]},
// CHECK:  {node_index: 23, kind: 'CodeBlock', text: '{', subtree_size: 21, children: [
// CHECK:    {node_index: 6, kind: 'IfStatement', text: 'if', has_error: yes, subtree_size: 4, children: [
// CHECK:      {node_index: 3, kind: 'NameReference', text: 'a'},
// CHECK:      {node_index: 5, kind: 'CodeBlock', text: '{', subtree_size: 2, children: [
// CHECK:        {node_index: 4, kind: 'CodeBlockEnd', text: '}'}]}]},
// CHECK:    {node_index: 11, kind: 'IfStatement', text: 'if', subtree_size: 5, children: [
// CHECK:      {node_index: 8, kind: 'Condition', text: '(', has_error: yes, subtree_size: 2, children: [
// CHECK:        {node_index: 7, kind: 'ConditionEnd', text: ')'}]},
// CHECK:      {node_index: 10, kind: 'CodeBlock', text: '{', subtree_size: 2, children: [
// CHECK:        {node_index: 9, kind: 'CodeBlockEnd', text: '}'}]}]},
// CHECK:    {node_index: 17, kind: 'IfStatement', text: 'if', subtree_size: 6, children: [
// CHECK:      {node_index: 14, kind: 'Condition', text: '(', has_error: yes, subtree_size: 3, children: [
// CHECK:        {node_index: 12, kind: 'NameReference', text: 'b'},
// CHECK:        {node_index: 13, kind: 'ConditionEnd', text: ')'}]},
// CHECK:      {node_index: 16, kind: 'CodeBlock', text: '{', subtree_size: 2, children: [
// CHECK:        {node_index: 15, kind: 'CodeBlockEnd', text: '}'}]}]},
// CHECK:    {node_index: 21, kind: 'IfStatement', text: 'if', has_error: yes, subtree_size: 4, children: [
// CHECK:      {node_index: 20, kind: 'Condition', text: '(', subtree_size: 3, children: [
// CHECK:        {node_index: 18, kind: 'NameReference', text: 'd'},
// CHECK:        {node_index: 19, kind: 'ConditionEnd', text: ')'}]}]},
// CHECK:    {node_index: 22, kind: 'CodeBlockEnd', text: '}'}]}]},
// CHECK:{node_index: 25, kind: 'FileEnd', text: ''},
// CHECK:]
}
