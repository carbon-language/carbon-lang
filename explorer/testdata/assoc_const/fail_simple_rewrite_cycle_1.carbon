// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

interface TwoTypes {
  let T:! Type;
  let U:! Type;
}

// Attempting to fully resolve the replacement for `.T` would never create a
// situation where `.T`'s expansion involves `.T`. Ensure we catch this anyway.
// CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/assoc_const/fail_simple_rewrite_cycle_1.carbon:[[@LINE+1]]: rewrite of (X).(TwoTypes.U) applies within its own resolved expansion of (X).(TwoTypes.U)*
fn F[X:! TwoTypes where .T = .U and .U = .U*]() {}

fn Main() -> i32 {
  return 0;
}
