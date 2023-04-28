// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

choice Opt {
  None(),
  Some(i32)
}

fn A(a: Opt, b: Opt) -> i32 {
  match ((a, b)) {
    case (Opt.Some(n: i32), _: auto) => { return n; }
    case (_: auto, Opt.Some(n: i32)) => { return n; }
    // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/match/fail_not_reachable.carbon:[[@LINE+1]]: unreachable case: all values matched by this case are matched by earlier cases
    case (Opt.Some(m: i32), Opt.Some(n: i32)) => { return m + n; }
  }
  return 0;
}

fn Main() -> i32 {
  return A(Opt.None(), Opt.Some(1));
}
