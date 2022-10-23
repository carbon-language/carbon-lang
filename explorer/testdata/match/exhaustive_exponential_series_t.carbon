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
  // This is T8 from http://moscova.inria.fr/~maranget/papers/warn/warn014.html
  match ((F(), F(), F(), F(), F(), F(), F(), F())) {
    case (AB.A(), AB.A(), AB.A(), AB.A(), AB.A(), AB.A(), AB.A(), AB.A()) => { return 0; }
    case (AB.B(), AB.B(), AB.B(), AB.B(), AB.B(), AB.B(), AB.B(), AB.B()) => { return 1; }
    case (_: AB,  AB.A(), AB.A(), AB.A(), AB.A(), AB.A(), AB.A(), AB.A()) => { return 2; }
    case (_: AB,  AB.B(), AB.B(), AB.B(), AB.B(), AB.B(), AB.B(), AB.B()) => { return 3; }
    case (_: AB,  _: AB,  AB.A(), AB.A(), AB.A(), AB.A(), AB.A(), AB.A()) => { return 4; }
    case (_: AB,  _: AB,  AB.B(), AB.B(), AB.B(), AB.B(), AB.B(), AB.B()) => { return 5; }
    case (_: AB,  _: AB,  _: AB,  AB.A(), AB.A(), AB.A(), AB.A(), AB.A()) => { return 6; }
    case (_: AB,  _: AB,  _: AB,  AB.B(), AB.B(), AB.B(), AB.B(), AB.B()) => { return 7; }
    case (_: AB,  _: AB,  _: AB,  _: AB,  AB.A(), AB.A(), AB.A(), AB.A()) => { return 8; }
    case (_: AB,  _: AB,  _: AB,  _: AB,  AB.B(), AB.B(), AB.B(), AB.B()) => { return 9; }
    case (_: AB,  _: AB,  _: AB,  _: AB,  _: AB,  AB.A(), AB.A(), AB.A()) => { return 10; }
    case (_: AB,  _: AB,  _: AB,  _: AB,  _: AB,  AB.B(), AB.B(), AB.B()) => { return 11; }
    case (_: AB,  _: AB,  _: AB,  _: AB,  _: AB,  _: AB,  AB.A(), AB.A()) => { return 12; }
    case (_: AB,  _: AB,  _: AB,  _: AB,  _: AB,  _: AB,  AB.B(), AB.B()) => { return 13; }
    case (_: AB,  _: AB,  _: AB,  _: AB,  _: AB,  _: AB,  _: AB,  AB.A()) => { return 14; }
    case (_: AB,  _: AB,  _: AB,  _: AB,  _: AB,  _: AB,  _: AB,  AB.B()) => { return 15; }
  }
}
