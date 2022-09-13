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

choice AB {
  A(),
  B()
}

fn F() -> AB { return AB.A(); }

fn Main() -> i32 {
  // Note, this match is exhaustive, but it exceeds our depth limit.
  match ((F(), F(), F(), F(), F(), F(), F(), F(), F())) {
    case (AB.A(), AB.A(), AB.A(), AB.A(), AB.A(), AB.A(), AB.A(), AB.A(), AB.A()) => { return 0; }
    case (AB.B(), AB.A(), AB.A(), AB.A(), AB.A(), AB.A(), AB.A(), AB.A(), AB.A()) => { return 1; }
    case (_: AB,  AB.B(), AB.A(), AB.A(), AB.A(), AB.A(), AB.A(), AB.A(), AB.A()) => { return 2; }
    case (_: AB,  _: AB,  AB.B(), AB.A(), AB.A(), AB.A(), AB.A(), AB.A(), AB.A()) => { return 3; }
    case (_: AB,  _: AB,  _: AB,  AB.B(), AB.A(), AB.A(), AB.A(), AB.A(), AB.A()) => { return 4; }
    case (_: AB,  _: AB,  _: AB,  _: AB,  AB.B(), AB.A(), AB.A(), AB.A(), AB.A()) => { return 5; }
    case (_: AB,  _: AB,  _: AB,  _: AB,  _: AB,  AB.B(), AB.A(), AB.A(), AB.A()) => { return 6; }
    case (_: AB,  _: AB,  _: AB,  _: AB,  _: AB,  _: AB,  AB.B(), AB.A(), AB.A()) => { return 7; }
    case (_: AB,  _: AB,  _: AB,  _: AB,  _: AB,  _: AB,  _: AB,  AB.B(), AB.A()) => { return 8; }
    case (_: AB,  _: AB,  _: AB,  _: AB,  _: AB,  _: AB,  _: AB,  _: AB,  AB.B()) => { return 9; }
  }
// CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/match/fail_exhaustive_exponential_9x.carbon:[[@LINE+1]]: non-exhaustive match may allow control-flow to reach the end of a function that provides a `->` return type
}
