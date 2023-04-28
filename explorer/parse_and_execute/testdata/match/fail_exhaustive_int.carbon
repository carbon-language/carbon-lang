// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

fn A(n: i32) -> i32 {
  match (n) {
    case 1 => { return 1; }
    case 2 => { return 2; }
  }
// CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/match/fail_exhaustive_int.carbon:[[@LINE+1]]: non-exhaustive match may allow control-flow to reach the end of a function that provides a `->` return type
}

fn Main() -> i32 {
  return A(0);
}
