// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{carbon} dump parse-tree %s 2>&1 | %{FileCheck-strict} %s
// AUTOUPDATE: %{carbon} dump parse-tree %s

// CHECK:{{.*}}/toolchain/parser/testdata/function/declaration/missing_name.carbon:[[@LINE+6]]:4: Expected function name after `fn` keyword.
// CHECK:[
// CHECK:{node_index: 1, kind: 'FunctionDeclaration', text: 'fn', has_error: yes, subtree_size: 2, children: [
// CHECK:  {node_index: 0, kind: 'DeclarationEnd', text: ';'}]},
// CHECK:{node_index: 2, kind: 'FileEnd', text: ''},
// CHECK:]
fn ();
