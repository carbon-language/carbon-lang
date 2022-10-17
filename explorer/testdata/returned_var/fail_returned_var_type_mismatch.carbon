// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

fn AddInt(a: i32, b: i32) -> i32 {
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/returned_var/fail_returned_var_type_mismatch.carbon:[[@LINE+1]]: type of returned var `bool` does not match return type `i32`
  returned var ret: bool = true;
  return var;
}

fn Main() -> i32 {
  return AddInt(1, 2);
}
