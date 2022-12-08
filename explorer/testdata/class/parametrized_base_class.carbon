// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: 2
// CHECK:STDOUT: 1
// CHECK:STDOUT: result: 0

package ExplorerTest api;

interface Number {
  fn Zero() -> Self;
  fn Add[self: Self](other: Self) -> Self;
}

external impl i32 as Number {
  fn Zero() -> i32 { return 0; }
  fn Add[self: i32](other: i32) -> i32 { return self + other; }
}

base class A(T:! Number) {
  var value_a: T;
}

class B extends A(i32) {
  var value_b: i32;
}

fn Main() -> i32 {
  var b: B = {.base = {.value_a = 1}, .value_b = 2};
  Print("{0}", b.value_b);
  Print("{0}", b.value_a);
  return 0;
}
