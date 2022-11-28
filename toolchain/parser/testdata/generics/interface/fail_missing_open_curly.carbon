// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{carbon-run-parser}
// CHECK:STDOUT: [
// CHECK:STDOUT:   {kind: 'DeclaredName', text: 'bar'},
// CHECK:STDOUT: {kind: 'InterfaceDefinition', text: 'interface', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:   {kind: 'DeclaredName', text: 'foo'},
// CHECK:STDOUT: {kind: 'InterfaceDefinition', text: 'interface', has_error: yes, subtree_size: 2},
// CHECK:STDOUT: {kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

// CHECK:STDERR: {{.*}}/toolchain/parser/testdata/generics/interface/fail_missing_open_curly.carbon:[[@LINE+1]]:15: Expected `{` to start interface definition.
interface bar baz {}

// CHECK:STDERR: {{.*}}/toolchain/parser/testdata/generics/interface/fail_missing_open_curly.carbon:[[@LINE+1]]:14: Expected `{` to start interface definition.
interface foo
