// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{carbon-run-parser}
// CHECK:STDOUT: [
// CHECK:STDOUT: {node_index: 51, kind: 'FunctionDefinition', text: '}', subtree_size: 52, children: [
// CHECK:STDOUT:   {node_index: 4, kind: 'FunctionDefinitionStart', text: '{', subtree_size: 5, children: [
// CHECK:STDOUT:     {node_index: 0, kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT:     {node_index: 1, kind: 'DeclaredName', text: 'F'},
// CHECK:STDOUT:     {node_index: 3, kind: 'ParameterList', text: '(', subtree_size: 2, children: [
// CHECK:STDOUT:       {node_index: 2, kind: 'ParameterListEnd', text: ')'}]}]},
// CHECK:STDOUT:   {node_index: 28, kind: 'IfStatement', text: 'if', subtree_size: 24, children: [
// CHECK:STDOUT:     {node_index: 7, kind: 'Condition', text: '(', subtree_size: 3, children: [
// CHECK:STDOUT:       {node_index: 5, kind: 'NameReference', text: 'a'},
// CHECK:STDOUT:       {node_index: 6, kind: 'ConditionEnd', text: ')'}]},
// CHECK:STDOUT:     {node_index: 22, kind: 'CodeBlock', text: 'if', has_error: yes, subtree_size: 15, children: [
// CHECK:STDOUT:       {node_index: 8, kind: 'CodeBlockStart', text: 'if', has_error: yes},
// CHECK:STDOUT:       {node_index: 21, kind: 'IfStatement', text: 'if', subtree_size: 13, children: [
// CHECK:STDOUT:         {node_index: 11, kind: 'Condition', text: '(', subtree_size: 3, children: [
// CHECK:STDOUT:           {node_index: 9, kind: 'NameReference', text: 'b'},
// CHECK:STDOUT:           {node_index: 10, kind: 'ConditionEnd', text: ')'}]},
// CHECK:STDOUT:         {node_index: 15, kind: 'CodeBlock', text: 'c', has_error: yes, subtree_size: 4, children: [
// CHECK:STDOUT:           {node_index: 12, kind: 'CodeBlockStart', text: 'c', has_error: yes},
// CHECK:STDOUT:           {node_index: 14, kind: 'ExpressionStatement', text: ';', subtree_size: 2, children: [
// CHECK:STDOUT:             {node_index: 13, kind: 'NameReference', text: 'c'}]}]},
// CHECK:STDOUT:         {node_index: 16, kind: 'IfStatementElse', text: 'else'},
// CHECK:STDOUT:         {node_index: 20, kind: 'CodeBlock', text: 'd', has_error: yes, subtree_size: 4, children: [
// CHECK:STDOUT:           {node_index: 17, kind: 'CodeBlockStart', text: 'd', has_error: yes},
// CHECK:STDOUT:           {node_index: 19, kind: 'ExpressionStatement', text: ';', subtree_size: 2, children: [
// CHECK:STDOUT:             {node_index: 18, kind: 'NameReference', text: 'd'}]}]}]}]},
// CHECK:STDOUT:     {node_index: 23, kind: 'IfStatementElse', text: 'else'},
// CHECK:STDOUT:     {node_index: 27, kind: 'CodeBlock', text: 'e', has_error: yes, subtree_size: 4, children: [
// CHECK:STDOUT:       {node_index: 24, kind: 'CodeBlockStart', text: 'e', has_error: yes},
// CHECK:STDOUT:       {node_index: 26, kind: 'ExpressionStatement', text: ';', subtree_size: 2, children: [
// CHECK:STDOUT:         {node_index: 25, kind: 'NameReference', text: 'e'}]}]}]},
// CHECK:STDOUT:   {node_index: 50, kind: 'IfStatement', text: 'if', subtree_size: 22, children: [
// CHECK:STDOUT:     {node_index: 31, kind: 'Condition', text: '(', subtree_size: 3, children: [
// CHECK:STDOUT:       {node_index: 29, kind: 'NameReference', text: 'x'},
// CHECK:STDOUT:       {node_index: 30, kind: 'ConditionEnd', text: ')'}]},
// CHECK:STDOUT:     {node_index: 35, kind: 'CodeBlock', text: '}', subtree_size: 4, children: [
// CHECK:STDOUT:       {node_index: 32, kind: 'CodeBlockStart', text: '{'},
// CHECK:STDOUT:       {node_index: 34, kind: 'ExpressionStatement', text: ';', subtree_size: 2, children: [
// CHECK:STDOUT:         {node_index: 33, kind: 'NameReference', text: 'f'}]}]},
// CHECK:STDOUT:     {node_index: 36, kind: 'IfStatementElse', text: 'else'},
// CHECK:STDOUT:     {node_index: 49, kind: 'IfStatement', text: 'if', subtree_size: 13, children: [
// CHECK:STDOUT:       {node_index: 39, kind: 'Condition', text: '(', subtree_size: 3, children: [
// CHECK:STDOUT:         {node_index: 37, kind: 'NameReference', text: 'x'},
// CHECK:STDOUT:         {node_index: 38, kind: 'ConditionEnd', text: ')'}]},
// CHECK:STDOUT:       {node_index: 43, kind: 'CodeBlock', text: '}', subtree_size: 4, children: [
// CHECK:STDOUT:         {node_index: 40, kind: 'CodeBlockStart', text: '{'},
// CHECK:STDOUT:         {node_index: 42, kind: 'ExpressionStatement', text: ';', subtree_size: 2, children: [
// CHECK:STDOUT:           {node_index: 41, kind: 'NameReference', text: 'g'}]}]},
// CHECK:STDOUT:       {node_index: 44, kind: 'IfStatementElse', text: 'else'},
// CHECK:STDOUT:       {node_index: 48, kind: 'CodeBlock', text: '}', subtree_size: 4, children: [
// CHECK:STDOUT:         {node_index: 45, kind: 'CodeBlockStart', text: '{'},
// CHECK:STDOUT:         {node_index: 47, kind: 'ExpressionStatement', text: ';', subtree_size: 2, children: [
// CHECK:STDOUT:           {node_index: 46, kind: 'NameReference', text: 'h'}]}]}]}]}]},
// CHECK:STDOUT: {node_index: 52, kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

fn F() {
  if (a)
    // CHECK:STDERR: {{.*}}/toolchain/parser/testdata/if/fail_else_unbraced.carbon:[[@LINE+1]]:5: Expected braced code block.
    if (b)
      // CHECK:STDERR: {{.*}}/toolchain/parser/testdata/if/fail_else_unbraced.carbon:[[@LINE+1]]:7: Expected braced code block.
      c;
    else
      // CHECK:STDERR: {{.*}}/toolchain/parser/testdata/if/fail_else_unbraced.carbon:[[@LINE+1]]:7: Expected braced code block.
      d;
  else
    // CHECK:STDERR: {{.*}}/toolchain/parser/testdata/if/fail_else_unbraced.carbon:[[@LINE+1]]:5: Expected braced code block.
    e;
  if (x) { f; }
  else if (x) { g; }
  else { h; }
}
