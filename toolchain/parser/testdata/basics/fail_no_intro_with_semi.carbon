// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{carbon-run-parser}
// CHECK:STDOUT: [
// CHECK:STDOUT: {node_index: 0, kind: 'EmptyDeclaration', text: ';', has_error: yes},
// CHECK:STDOUT: {node_index: 1, kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

// CHECK:STDERR: {{.*}}/toolchain/parser/testdata/basics/fail_no_intro_with_semi.carbon:[[@LINE+1]]:1: Unrecognized declaration introducer.
foo;
