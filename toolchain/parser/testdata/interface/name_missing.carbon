// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{carbon-run-parser}
// CHECK:STDOUT: [
// CHECK:STDOUT: {node_index: 2, kind: 'InterfaceDeclaration', text: 'interface', has_error: yes, subtree_size: 3, children: [
// CHECK:STDOUT:   {node_index: 1, kind: 'InterfaceBody', text: '{', subtree_size: 2, children: [
// CHECK:STDOUT:     {node_index: 0, kind: 'InterfaceBodyEnd', text: '}'}]}]},
// CHECK:STDOUT: {node_index: 3, kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

// CHECK:STDERR: {{.*}}/toolchain/parser/testdata/interface/name_missing.carbon:[[@LINE+1]]:11: Expected function name after `interface` keyword.
interface {
}
