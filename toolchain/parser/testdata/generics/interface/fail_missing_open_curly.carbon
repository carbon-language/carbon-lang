// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: [
// CHECK:STDOUT:   {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:   {kind: 'DeclaredName', text: 'Bar'},
// CHECK:STDOUT: {kind: 'InterfaceDeclaration', text: 'interface', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:   {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:   {kind: 'DeclaredName', text: 'Foo'},
// CHECK:STDOUT: {kind: 'InterfaceDeclaration', text: 'interface', has_error: yes, subtree_size: 3},
// CHECK:STDOUT: {kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

// CHECK:STDERR: toolchain/parser/testdata/generics/interface/fail_missing_open_curly.carbon:[[@LINE+1]]:15: `interface` should either end with a `;` for a declaration or have a `{ ... }` block for a definition.
interface Bar Baz {}

// CHECK:STDERR: toolchain/parser/testdata/generics/interface/fail_missing_open_curly.carbon:[[@LINE+1]]:14: `interface` should either end with a `;` for a declaration or have a `{ ... }` block for a definition.
interface Foo
