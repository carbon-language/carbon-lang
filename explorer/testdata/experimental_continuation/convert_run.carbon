// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 3

package ExplorerTest api;

class Wrap(T:! type) {
  var v: T;
}
fn MakeWrap[T:! type](x: T) -> Wrap(T) { return {.v = x}; }

impl forall [T:! type] Wrap(T) as ImplicitAs(T) {
  fn Convert[self: Self]() -> T {
    return self.v;
  }
}

fn Main() -> i32 {
  var n: i32 = 1;
  __continuation k {
    n = n + 1;
  }
  n = 2;
  __run MakeWrap(k);
  return n;
}
