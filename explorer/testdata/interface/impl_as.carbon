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

interface Apple(T:! Type) {
  fn F[me: Self]() -> T;
}

interface Banana {
  impl as Apple(i32);
  fn G[me: Self]();
}

class Carrot {
  var n: i32;
}

external impl Carrot as Apple(i32) {
  fn F[me: Self]() -> i32 { return me.n; }
}

external impl Carrot as Banana {
  fn G[me: Self]() { Print("Carrot.G"); }
}

fn H[T:! Banana](x: T) -> i32 {
  x.G();
  return x.(Apple(i32).F)();
}

fn Main() -> i32 {
  var c: Carrot = {.n = 5};
  return H(c);
}
