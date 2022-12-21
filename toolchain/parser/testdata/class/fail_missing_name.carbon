// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{carbon-run-parser}
// CHECK:STDOUT: [
// CHECK:STDOUT:   {kind: 'DeclaredName', text: 'class', has_error: yes},
// CHECK:STDOUT:     {kind: 'ClassBodyStart', text: '{'},
// CHECK:STDOUT:   {kind: 'ClassBody', text: '}', subtree_size: 2},
// CHECK:STDOUT: {kind: 'ClassDefinition', text: 'class', has_error: yes, subtree_size: 4},
// CHECK:STDOUT: {kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

// CHECK:STDERR: {{.*}}/toolchain/parser/testdata/class/fail_missing_name.carbon:[[@LINE+1]]:7: Expected identifier after `class` keyword.
class {
}
