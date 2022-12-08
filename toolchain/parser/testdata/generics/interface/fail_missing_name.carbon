// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{carbon-run-parser}
// CHECK:STDOUT: [
// CHECK:STDOUT:   {kind: 'DeclaredName', text: 'interface', has_error: yes},
// CHECK:STDOUT:     {kind: 'InterfaceBodyStart', text: '{'},
// CHECK:STDOUT:   {kind: 'InterfaceBody', text: '}', subtree_size: 2},
// CHECK:STDOUT: {kind: 'InterfaceDefinition', text: 'interface', has_error: yes, subtree_size: 4},
// CHECK:STDOUT: {kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

// CHECK:STDERR: {{.*}}/toolchain/parser/testdata/generics/interface/fail_missing_name.carbon:[[@LINE+1]]:11: Expected interface name after `interface` keyword.
interface {
}
