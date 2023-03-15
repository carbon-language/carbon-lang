// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{carbon-run-parser}
// CHECK:STDOUT: [
// CHECK:STDOUT:   {kind: 'ClassIntroducer', text: 'class'},
// CHECK:STDOUT:   {kind: 'DeclaredName', text: 'Foo'},
// CHECK:STDOUT: {kind: 'ClassDeclaration', text: ';', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:   {kind: 'InterfaceIntroducer', text: 'interface'},
// CHECK:STDOUT:   {kind: 'DeclaredName', text: 'Bar'},
// CHECK:STDOUT: {kind: 'InterfaceDeclaration', text: 'interface', has_error: yes, subtree_size: 3},
// CHECK:STDOUT: {kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

// CHECK:STDERR: {{.*}}/toolchain/parser/testdata/generics/deduced_params/two_suffix_comma.carbon:[[@LINE+1]]:10: `class` should either end with a `;` for a declaration or have a `{ ... }` block for a definition.
class Foo[a: i32, b: i32,]();

// CHECK:STDERR: {{.*}}/toolchain/parser/testdata/generics/deduced_params/two_suffix_comma.carbon:[[@LINE+1]]:14: `interface` should either end with a `;` for a declaration or have a `{ ... }` block for a definition.
interface Bar[a: i32, b: i32,]() {}
