// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 0

package ExplorerTest api;

__mixin Operations {
  fn Square[me: Self](x:i32) -> i32{
    return x * x;
  }
}

class Point {
  fn Origin() -> Point {
    return {.x = 0, .y = 0};
  }

  var x: i32;
  var y: i32;
  __mix Operations;
}

class Complex {
  fn Zero() -> Complex {
    return {.r = 0, .i = 0};
  }

  var r: i32;
  var i: i32;
  __mix Operations;
}


fn Main() -> i32 {
  var p: Point = Point.Origin();
  var c: Complex = Complex.Zero();
  p.x = 3;
  c.r = 4;
  return p.Square(p.x) - c.Square(c.r) + 7;
}
