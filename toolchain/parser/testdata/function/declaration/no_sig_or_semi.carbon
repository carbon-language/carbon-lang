// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{carbon} dump parse-tree %s 2>&1 | %{FileCheck-strict} %s
// AUTOUPDATE: %{carbon} dump parse-tree %s

// CHECK:{{.*}}/toolchain/parser/testdata/function/declaration/no_sig_or_semi.carbon:[[@LINE+6]]:7: Expected `(` after function name.
// CHECK:[
// CHECK:{node_index: 1, kind: 'FunctionDeclaration', text: 'fn', has_error: yes, subtree_size: 2, children: [
// CHECK:  {node_index: 0, kind: 'DeclaredName', text: 'foo'}]},
// CHECK:{node_index: 2, kind: 'FileEnd', text: ''},
// CHECK:]
fn foo
