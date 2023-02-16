// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: Carrot.G
// CHECK:STDOUT: result: 5

package ExplorerTest api;

interface Apple(T:! type) {
  fn F[self: Self]() -> T;
}

interface Banana {
  extends Apple(i32);
  fn G[self: Self]();
}

class Carrot {
  var n: i32;
}

// This impl also provides an `impl Carrot as Apple(i32)`.
external impl Carrot as Banana {
  fn F[self: Self]() -> i32 { return self.n; }
  fn G[self: Self]() { Print("Carrot.G"); }
}

fn Main() -> i32 {
  var c: Carrot = {.n = 5};
  c.(Banana.G)();
  return c.(Apple(i32).F)();
}
