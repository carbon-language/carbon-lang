// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 12

package ExplorerTest api;

interface SwizzleWith(T:! type) {
  fn Op[self: Self](x: T) -> i32;
}

impl i32 as SwizzleWith(i32) {
  fn Op[self: Self](x: Self) -> Self { return self * 10 + x; }
}

fn F[T:! SwizzleWith(.Self)](v: T, w: T) -> i32 {
  return v.Op(w);
}

fn Main() -> i32 {
  var one: i32 = 1;
  var two: i32 = 2;
  return F(one, two);
}
