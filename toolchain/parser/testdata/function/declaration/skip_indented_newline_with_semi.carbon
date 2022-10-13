// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{carbon} dump parse-tree %s | %{FileCheck-strict} %s
// AUTOUPDATE: %{carbon} dump parse-tree %s

// CHECK:{{.*}}/toolchain/parser/testdata/function/declaration/skip_indented_newline_with_semi.carbon:[[@LINE+11]]:4: Expected function name after `fn` keyword.
// CHECK:[
// CHECK:{node_index: 1, kind: 'FunctionDeclaration', text: 'fn', has_error: yes, subtree_size: 2, children: [
// CHECK:  {node_index: 0, kind: 'DeclarationEnd', text: ';'}]},
// CHECK:{node_index: 6, kind: 'FunctionDeclaration', text: 'fn', subtree_size: 5, children: [
// CHECK:  {node_index: 2, kind: 'DeclaredName', text: 'F'},
// CHECK:  {node_index: 4, kind: 'ParameterList', text: '(', subtree_size: 2, children: [
// CHECK:    {node_index: 3, kind: 'ParameterListEnd', text: ')'}]},
// CHECK:  {node_index: 5, kind: 'DeclarationEnd', text: ';'}]},
// CHECK:{node_index: 7, kind: 'FileEnd', text: ''},
// CHECK:]
fn (x,
    y,
    z);
fn F();
