// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

fn AddInt(a: i32, b: i32) -> i32 {
  returned var ret: i32 = a + b;
  if (a == b) {
    // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/returned_var/fail_duplicate_return_var.carbon:[[@LINE+1]]: Duplicate definition of returned var also found at {{.*}}/explorer/testdata/returned_var/fail_duplicate_return_var.carbon:[[@LINE-2]]
    returned var ret2: i32 = a + b;
  }
  return var;
}

fn Main() -> i32 {
  return AddInt(1, 2);
}
