// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{carbon-run-parser}
// CHECK:STDOUT: [
// CHECK:STDOUT: {node_index: 27, kind: 'FunctionDeclaration', text: 'fn', subtree_size: 28, children: [
// CHECK:STDOUT:   {node_index: 0, kind: 'DeclaredName', text: 'F'},
// CHECK:STDOUT:   {node_index: 2, kind: 'ParameterList', text: '(', subtree_size: 2, children: [
// CHECK:STDOUT:     {node_index: 1, kind: 'ParameterListEnd', text: ')'}]},
// CHECK:STDOUT:   {node_index: 26, kind: 'CodeBlock', text: '{', subtree_size: 24, children: [
// CHECK:STDOUT:     {node_index: 24, kind: 'WhileStatement', text: 'while', subtree_size: 22, children: [
// CHECK:STDOUT:       {node_index: 5, kind: 'Condition', text: '(', subtree_size: 3, children: [
// CHECK:STDOUT:         {node_index: 3, kind: 'NameReference', text: 'a'},
// CHECK:STDOUT:         {node_index: 4, kind: 'ConditionEnd', text: ')'}]},
// CHECK:STDOUT:       {node_index: 23, kind: 'CodeBlock', text: '{', subtree_size: 18, children: [
// CHECK:STDOUT:         {node_index: 13, kind: 'IfStatement', text: 'if', subtree_size: 8, children: [
// CHECK:STDOUT:           {node_index: 8, kind: 'Condition', text: '(', subtree_size: 3, children: [
// CHECK:STDOUT:             {node_index: 6, kind: 'NameReference', text: 'b'},
// CHECK:STDOUT:             {node_index: 7, kind: 'ConditionEnd', text: ')'}]},
// CHECK:STDOUT:           {node_index: 12, kind: 'CodeBlock', text: '{', subtree_size: 4, children: [
// CHECK:STDOUT:             {node_index: 10, kind: 'BreakStatement', text: 'break', subtree_size: 2, children: [
// CHECK:STDOUT:               {node_index: 9, kind: 'StatementEnd', text: ';'}]},
// CHECK:STDOUT:             {node_index: 11, kind: 'CodeBlockEnd', text: '}'}]}]},
// CHECK:STDOUT:         {node_index: 21, kind: 'IfStatement', text: 'if', subtree_size: 8, children: [
// CHECK:STDOUT:           {node_index: 16, kind: 'Condition', text: '(', subtree_size: 3, children: [
// CHECK:STDOUT:             {node_index: 14, kind: 'NameReference', text: 'c'},
// CHECK:STDOUT:             {node_index: 15, kind: 'ConditionEnd', text: ')'}]},
// CHECK:STDOUT:           {node_index: 20, kind: 'CodeBlock', text: '{', subtree_size: 4, children: [
// CHECK:STDOUT:             {node_index: 18, kind: 'ContinueStatement', text: 'continue', subtree_size: 2, children: [
// CHECK:STDOUT:               {node_index: 17, kind: 'StatementEnd', text: ';'}]},
// CHECK:STDOUT:             {node_index: 19, kind: 'CodeBlockEnd', text: '}'}]}]},
// CHECK:STDOUT:         {node_index: 22, kind: 'CodeBlockEnd', text: '}'}]}]},
// CHECK:STDOUT:     {node_index: 25, kind: 'CodeBlockEnd', text: '}'}]}]},
// CHECK:STDOUT: {node_index: 28, kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

fn F() {
  while (a) {
    if (b) {
      break;
    }
    if (c) {
      continue;
    }
  }
}
