// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | %{FileCheck-strict} %s
// RUN: %{not} %{explorer-trace} %s 2>&1 | %{FileCheck-allow-unmatched} %s
// AUTOUPDATE: %{explorer} %s

package ExplorerTest api;

fn AddInt(a: i32, b: i32) -> i32 {
  returned var ret: i32 = a + b;
  if (a == b) {
    // CHECK:COMPILATION ERROR: {{.*}}/explorer/testdata/returned_var/fail_duplicate_return_var.carbon:[[@LINE+1]]: Duplicate definition of returned var also found at {{.*}}/explorer/testdata/returned_var/fail_duplicate_return_var.carbon:12
    returned var ret2: i32 = a + b;
  }
  return var;
}

fn Main() -> i32 {
  return AddInt(1, 2);
}
