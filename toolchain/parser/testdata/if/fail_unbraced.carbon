// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// TODO: This should have an error.
// AUTOUPDATE
// RUN: %{not} %{carbon-run-parser}
// CHECK:STDOUT: [
// CHECK:STDOUT: {node_index: 25, kind: 'FunctionDefinition', text: '}', subtree_size: 26, children: [
// CHECK:STDOUT:   {node_index: 4, kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5, children: [
// CHECK:STDOUT:     {node_index: 0, kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:     {node_index: 1, kind: 'DeclaredName', text: 'F'},
// CHECK:STDOUT:     {node_index: 3, kind: 'ParameterList', text: '(', subtree_size: 2, children: [
// CHECK:STDOUT:       {node_index: 2, kind: 'ParameterListEnd', text: ')'}]}]},
// CHECK:STDOUT:   {node_index: 24, kind: 'IfStatement', text: 'if', subtree_size: 20, children: [
// CHECK:STDOUT:     {node_index: 7, kind: 'Condition', text: '(', subtree_size: 3, children: [
// CHECK:STDOUT:       {node_index: 5, kind: 'NameReference', text: 'a'},
// CHECK:STDOUT:       {node_index: 6, kind: 'ConditionEnd', text: ')'}]},
// CHECK:STDOUT:     {node_index: 23, kind: 'CodeBlock', text: 'if', has_error: yes, subtree_size: 16, children: [
// CHECK:STDOUT:       {node_index: 8, kind: 'CodeBlockStart', text: 'if', has_error: yes},
// CHECK:STDOUT:       {node_index: 22, kind: 'IfStatement', text: 'if', subtree_size: 14, children: [
// CHECK:STDOUT:         {node_index: 11, kind: 'Condition', text: '(', subtree_size: 3, children: [
// CHECK:STDOUT:           {node_index: 9, kind: 'NameReference', text: 'b'},
// CHECK:STDOUT:           {node_index: 10, kind: 'ConditionEnd', text: ')'}]},
// CHECK:STDOUT:         {node_index: 21, kind: 'CodeBlock', text: 'if', has_error: yes, subtree_size: 10, children: [
// CHECK:STDOUT:           {node_index: 12, kind: 'CodeBlockStart', text: 'if', has_error: yes},
// CHECK:STDOUT:           {node_index: 20, kind: 'IfStatement', text: 'if', subtree_size: 8, children: [
// CHECK:STDOUT:             {node_index: 15, kind: 'Condition', text: '(', subtree_size: 3, children: [
// CHECK:STDOUT:               {node_index: 13, kind: 'NameReference', text: 'c'},
// CHECK:STDOUT:               {node_index: 14, kind: 'ConditionEnd', text: ')'}]},
// CHECK:STDOUT:             {node_index: 19, kind: 'CodeBlock', text: 'd', has_error: yes, subtree_size: 4, children: [
// CHECK:STDOUT:               {node_index: 16, kind: 'CodeBlockStart', text: 'd', has_error: yes},
// CHECK:STDOUT:               {node_index: 18, kind: 'ExpressionStatement', text: ';', subtree_size: 2, children: [
// CHECK:STDOUT:                 {node_index: 17, kind: 'NameReference', text: 'd'}]}]}]}]}]}]}]}]},
// CHECK:STDOUT: {node_index: 26, kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

fn F() {
  if (a)
    // CHECK:STDERR: {{.*}}/toolchain/parser/testdata/if/fail_unbraced.carbon:[[@LINE+1]]:5: Expected braced code block.
    if (b)
      // CHECK:STDERR: {{.*}}/toolchain/parser/testdata/if/fail_unbraced.carbon:[[@LINE+1]]:7: Expected braced code block.
      if (c)
        // CHECK:STDERR: {{.*}}/toolchain/parser/testdata/if/fail_unbraced.carbon:[[@LINE+1]]:9: Expected braced code block.
        d;
}
