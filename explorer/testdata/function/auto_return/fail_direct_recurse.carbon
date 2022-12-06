// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

// This is required to fail even though the Recurse() call's return value isn't
// used.
fn Recurse(x: i32, do_recurse: bool) -> auto {
  if (do_recurse) {
    // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/function/auto_return/fail_direct_recurse.carbon:[[@LINE+1]]: Function calls itself, but has a deduced return type
    Recurse(x, false);
  }
  return x;
}

fn Main() -> i32 {
  return Recurse(1, true) - 3;
}
