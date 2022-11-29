// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

fn AddInt(a: i32, b: i32) -> auto {
  return a + b;
}

fn Main() -> i32 {
  var x: i32;
  if (0 == 1) {
    x = 0;
  }
  // Static analysis thinks `x` may be formed, defer the check to run-time.
  // CHECK:STDERR: RUNTIME ERROR: {{.*}}/explorer/testdata/unformed/dynamic/fail_param.carbon:[[@LINE+1]]: undefined behavior: access to uninitialized value Uninit<Placeholder<x>>
  return AddInt(x, 2);
}
