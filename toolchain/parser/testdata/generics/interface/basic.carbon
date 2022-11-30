// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{carbon-run-parser}
// CHECK:STDOUT: [
// CHECK:STDOUT:   {kind: 'DeclaredName', text: 'foo'},
// CHECK:STDOUT:     {kind: 'InterfaceBodyStart', text: '{'},
// CHECK:STDOUT:   {kind: 'InterfaceBody', text: '}', subtree_size: 2},
// CHECK:STDOUT: {kind: 'InterfaceDefinition', text: 'interface', subtree_size: 4},
// CHECK:STDOUT: {kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]
interface foo {
}
