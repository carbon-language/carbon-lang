// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 1

package ExplorerTest api;

class Point(T:! Type, V:! T) {
  fn Get[me: Self]() -> T {
    return V;
  }
}

fn F(p: Point(i32, 1)) -> i32 {
  return p.Get();
}

fn Main() -> i32 {
  var v: Point(i32, 1) = {};
  return F(v);
}
