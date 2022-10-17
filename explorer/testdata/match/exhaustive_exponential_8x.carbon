// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 0

package ExplorerTest api;

choice AB {
  A(),
  B()
}

fn F() -> AB { return AB.A(); }

fn Main() -> i32 {
  match ((F(), F(), F(), F(), F(), F(), F(), F())) {
    case (AB.A(), AB.A(), AB.A(), AB.A(), AB.A(), AB.A(), AB.A(), AB.A()) => { return 0; }
    case (AB.B(), AB.A(), AB.A(), AB.A(), AB.A(), AB.A(), AB.A(), AB.A()) => { return 1; }
    case (_: AB,  AB.B(), AB.A(), AB.A(), AB.A(), AB.A(), AB.A(), AB.A()) => { return 2; }
    case (_: AB,  _: AB,  AB.B(), AB.A(), AB.A(), AB.A(), AB.A(), AB.A()) => { return 3; }
    case (_: AB,  _: AB,  _: AB,  AB.B(), AB.A(), AB.A(), AB.A(), AB.A()) => { return 4; }
    case (_: AB,  _: AB,  _: AB,  _: AB,  AB.B(), AB.A(), AB.A(), AB.A()) => { return 5; }
    case (_: AB,  _: AB,  _: AB,  _: AB,  _: AB,  AB.B(), AB.A(), AB.A()) => { return 6; }
    case (_: AB,  _: AB,  _: AB,  _: AB,  _: AB,  _: AB,  AB.B(), AB.A()) => { return 7; }
    case (_: AB,  _: AB,  _: AB,  _: AB,  _: AB,  _: AB,  _: AB,  AB.B()) => { return 8; }
  }
}
