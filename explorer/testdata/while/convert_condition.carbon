// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 0

package ExplorerTest api;

class LazyNe {
  var v1: i32;
  var v2: i32;
  impl as ImplicitAs(bool) {
    fn Convert[self: Self]() -> bool {
      return not (self.v1 == self.v2);
    }
  }
  fn Make(v1: i32, v2: i32) -> Self {
    return {.v1 = v1, .v2 = v2};
  }
}

fn Main() -> i32 {
  var x: auto = 2;
  while (LazyNe.Make(x, 0)) {
    x = x - 1;
  }
  return x;
}
