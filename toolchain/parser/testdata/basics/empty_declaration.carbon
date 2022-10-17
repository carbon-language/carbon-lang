// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{carbon-run-parser}
// CHECK:STDOUT: [
// CHECK:STDOUT: {node_index: 0, kind: 'EmptyDeclaration', text: ';'},
// CHECK:STDOUT: {node_index: 1, kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

;
