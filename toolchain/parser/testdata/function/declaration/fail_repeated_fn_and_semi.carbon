// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{carbon-run-parser}
// CHECK:STDOUT: [
// CHECK:STDOUT:   {kind: 'FunctionIntroducer', text: 'fn'},
// CHECK:STDOUT: {kind: 'FunctionDeclaration', text: ';', has_error: yes, subtree_size: 2},
// CHECK:STDOUT: {kind: 'FileEnd', text: ''},
// CHECK:STDOUT: ]

// CHECK:STDERR: {{.*}}/toolchain/parser/testdata/function/declaration/fail_repeated_fn_and_semi.carbon:[[@LINE+1]]:4: Expected function name after `fn` keyword.
fn fn;
