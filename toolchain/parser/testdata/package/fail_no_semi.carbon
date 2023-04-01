// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{carbon-run-parser}
// CHECK:STDOUT: [
// CHECK:STDOUT:   {kind: 'PackageIntroducer', text: 'package'},
// CHECK:STDOUT:   {kind: 'DeclaredName', text: 'Geometry'},
// CHECK:STDOUT:   {kind: 'PackageApi', text: 'api'},
// CHECK:STDOUT: {kind: 'PackageDirective', text: 'package', has_error: yes, subtree_size: 4},
// CHECK:STDOUT: {kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

// CHECK:STDERR: {{.*}}/toolchain/parser/testdata/package/fail_no_semi.carbon:[[@LINE+1]]:21: Expected `;` to end package directive.
package Geometry api
