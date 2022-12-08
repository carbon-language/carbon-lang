// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{carbon-run-parser}
// CHECK:STDOUT: [
// CHECK:STDOUT:   {kind: 'PackageIntroducer', text: 'package'},
// CHECK:STDOUT:   {kind: 'DeclaredName', text: 'Foo'},
// CHECK:STDOUT:     {kind: 'Literal', text: '"bar"'},
// CHECK:STDOUT:   {kind: 'PackageLibrary', text: 'library', subtree_size: 2},
// CHECK:STDOUT: {kind: 'PackageDirective', text: ';', has_error: yes, subtree_size: 5},
// CHECK:STDOUT: {kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

// CHECK:STDERR: {{.*}}/toolchain/parser/testdata/package/fail_extra_string.carbon:[[@LINE+1]]:27: Expected a `api` or `impl`.
package Foo library "bar" "baz";
