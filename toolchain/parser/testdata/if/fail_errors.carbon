// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{carbon-run-parser}
// CHECK:STDOUT: [
// CHECK:STDOUT: {node_index: 26, kind: 'FunctionDefinition', text: '}', subtree_size: 27, children: [
// CHECK:STDOUT:   {node_index: 4, kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5, children: [
// CHECK:STDOUT:     {node_index: 0, kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:     {node_index: 1, kind: 'DeclaredName', text: 'F'},
// CHECK:STDOUT:     {node_index: 3, kind: 'ParameterList', text: '(', subtree_size: 2, children: [
// CHECK:STDOUT:       {node_index: 2, kind: 'ParameterListEnd', text: ')'}]}]},
// CHECK:STDOUT:   {node_index: 8, kind: 'IfStatement', text: 'if', has_error: yes, subtree_size: 4, children: [
// CHECK:STDOUT:     {node_index: 5, kind: 'NameReference', text: 'a'},
// CHECK:STDOUT:     {node_index: 7, kind: 'CodeBlock', text: '}', subtree_size: 2, children: [
// CHECK:STDOUT:       {node_index: 6, kind: 'CodeBlockStart', text: '{'}]}]},
// CHECK:STDOUT:   {node_index: 13, kind: 'IfStatement', text: 'if', subtree_size: 5, children: [
// CHECK:STDOUT:     {node_index: 10, kind: 'Condition', text: '(', has_error: yes, subtree_size: 2, children: [
// CHECK:STDOUT:       {node_index: 9, kind: 'ConditionEnd', text: ')'}]},
// CHECK:STDOUT:     {node_index: 12, kind: 'CodeBlock', text: '}', subtree_size: 2, children: [
// CHECK:STDOUT:       {node_index: 11, kind: 'CodeBlockStart', text: '{'}]}]},
// CHECK:STDOUT:   {node_index: 19, kind: 'IfStatement', text: 'if', subtree_size: 6, children: [
// CHECK:STDOUT:     {node_index: 16, kind: 'Condition', text: '(', has_error: yes, subtree_size: 3, children: [
// CHECK:STDOUT:       {node_index: 14, kind: 'NameReference', text: 'b'},
// CHECK:STDOUT:       {node_index: 15, kind: 'ConditionEnd', text: ')'}]},
// CHECK:STDOUT:     {node_index: 18, kind: 'CodeBlock', text: '}', subtree_size: 2, children: [
// CHECK:STDOUT:       {node_index: 17, kind: 'CodeBlockStart', text: '{'}]}]},
// CHECK:STDOUT:   {node_index: 25, kind: 'IfStatement', text: 'if', subtree_size: 6, children: [
// CHECK:STDOUT:     {node_index: 22, kind: 'Condition', text: '(', subtree_size: 3, children: [
// CHECK:STDOUT:       {node_index: 20, kind: 'NameReference', text: 'd'},
// CHECK:STDOUT:       {node_index: 21, kind: 'ConditionEnd', text: ')'}]},
// CHECK:STDOUT:     {node_index: 24, kind: 'CodeBlock', text: '}', has_error: yes, subtree_size: 2, children: [
// CHECK:STDOUT:       {node_index: 23, kind: 'CodeBlockStart', text: '}', has_error: yes}]}]}]},
// CHECK:STDOUT: {node_index: 27, kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

fn F() {
  // CHECK:STDERR: {{.*}}/toolchain/parser/testdata/if/fail_errors.carbon:[[@LINE+1]]:6: Expected `(` after `if`.
  if a {}
  // CHECK:STDERR: {{.*}}/toolchain/parser/testdata/if/fail_errors.carbon:[[@LINE+1]]:7: Expected expression.
  if () {}
  // CHECK:STDERR: {{.*}}/toolchain/parser/testdata/if/fail_errors.carbon:[[@LINE+1]]:9: Unexpected tokens before `)`.
  if (b c) {}
  if (d)
// CHECK:STDERR: {{.*}}/toolchain/parser/testdata/if/fail_errors.carbon:[[@LINE+2]]:1: Expected braced code block.
// CHECK:STDERR: {{.*}}/toolchain/parser/testdata/if/fail_errors.carbon:[[@LINE+1]]:1: Expected expression.
}
