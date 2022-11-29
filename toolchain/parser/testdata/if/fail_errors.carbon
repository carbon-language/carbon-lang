// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{carbon-run-parser}
// CHECK:STDOUT: [
// CHECK:STDOUT:     {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:     {kind: 'DeclaredName', text: 'F'},
// CHECK:STDOUT:       {kind: 'ParameterListStart', text: '('},
// CHECK:STDOUT:     {kind: 'ParameterList', text: ')', subtree_size: 2},
// CHECK:STDOUT:   {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'IfConditionStart', text: 'if', has_error: yes},
// CHECK:STDOUT:       {kind: 'NameReference', text: 'a'},
// CHECK:STDOUT:     {kind: 'IfCondition', text: 'if', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:       {kind: 'CodeBlockStart', text: '{'},
// CHECK:STDOUT:     {kind: 'CodeBlock', text: '}', subtree_size: 2},
// CHECK:STDOUT:   {kind: 'IfStatement', text: 'if', subtree_size: 6},
// CHECK:STDOUT:       {kind: 'IfConditionStart', text: '('},
// CHECK:STDOUT:     {kind: 'IfCondition', text: ')', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:       {kind: 'CodeBlockStart', text: '{'},
// CHECK:STDOUT:     {kind: 'CodeBlock', text: '}', subtree_size: 2},
// CHECK:STDOUT:   {kind: 'IfStatement', text: 'if', subtree_size: 5},
// CHECK:STDOUT:       {kind: 'IfConditionStart', text: '('},
// CHECK:STDOUT:       {kind: 'NameReference', text: 'b'},
// CHECK:STDOUT:     {kind: 'IfCondition', text: ')', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:       {kind: 'CodeBlockStart', text: '{'},
// CHECK:STDOUT:     {kind: 'CodeBlock', text: '}', subtree_size: 2},
// CHECK:STDOUT:   {kind: 'IfStatement', text: 'if', subtree_size: 6},
// CHECK:STDOUT:       {kind: 'IfConditionStart', text: '('},
// CHECK:STDOUT:       {kind: 'NameReference', text: 'd'},
// CHECK:STDOUT:     {kind: 'IfCondition', text: ')', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'CodeBlockStart', text: '}', has_error: yes},
// CHECK:STDOUT:     {kind: 'CodeBlock', text: '}', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:   {kind: 'IfStatement', text: 'if', subtree_size: 6},
// CHECK:STDOUT: {kind: 'FunctionDefinition', text: '}', subtree_size: 29},
// CHECK:STDOUT: {kind: 'FileEnd', text: ''},
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
