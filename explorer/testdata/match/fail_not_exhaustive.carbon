// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{not} %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s

package ExplorerTest api;

choice Opt {
  None(),
  Some(i32)
}

fn A(a: Opt, b: Opt) -> i32 {
  match ((a, b)) {
    case (Opt.Some(n: i32), _: auto) => { return n; }
    case (_: auto, Opt.Some(n: i32)) => { return n; }
  }
// CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/match/fail_not_exhaustive.carbon:[[@LINE+1]]: non-exhaustive match may allow control-flow to reach the end of a function that provides a `->` return type
}

fn Main() -> i32 {
  return A(Opt.None(), Opt.Some(1));
}
