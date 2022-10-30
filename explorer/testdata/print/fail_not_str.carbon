// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

fn Main() -> i32 {
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/print/fail_not_str.carbon:[[@LINE+3]]: type error in Print argument 0
  // CHECK:STDERR: expected: String
  // CHECK:STDERR: actual: {.x: i32}
  Print({.x = 1});
  return 0;
}
