// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 0

package ExplorerTest api;

interface MyAddWith(T:! Type) {
  fn Op[me: Self](b: T) -> Self;
}

class Point {
  var x: i32;
  var y: i32;
}

// Allowed: `Self` means `Point` after `as`
impl Point as MyAddWith(Self) {
  fn Op[me: Point](b: Point) -> Point {
    return {.x = me.x + b.x, .y = me.y + b.y};
  }
}

fn DoAddGeneric[T:! Type, U:! MyAddWith(T)](a: U, b: T) -> U {
  return a.Op(b);
}

fn Main() -> i32 {
  var a: Point = {.x = 2, .y = 1};
  var b: Point = {.x = 4, .y = 2};
  var p: Point = DoAddGeneric(a, b);
  return p.x - p.y - 3;
}
