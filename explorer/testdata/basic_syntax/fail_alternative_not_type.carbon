// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

// CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/basic_syntax/fail_alternative_not_type.carbon:[[@LINE+1]]: type error in type expression: '(i32)' is not implicitly convertible to 'Type'
choice C { X(42) }

fn Main() -> i32 {
  return 0;
}
