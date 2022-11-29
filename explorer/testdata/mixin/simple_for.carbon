// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | %{FileCheck-strict} %s
// RUN: %{explorer-trace} %s 2>&1 | %{FileCheck-allow-unmatched} %s
// AUTOUPDATE: %{explorer} %s
// CHECK:result: 0

package ExplorerTest api;

interface Multipliable {
  fn Mul[me: Self](other: Self) -> Self;
}

__mixin Operations for Multipliable {
  export fn Square[me: Self]() -> Self {
    return me.Mul(me);
  }
}

class Point {
  fn Origin() -> Point {
    return {.x = 0, .y = 0};
  }

  __mix Operations;

  impl as Multipliable {
    fn Mul[me: Self](other: Self) -> Self {
       return {.x = me.x * other.x, .y = me.y * other.y};
    }
  }

  var x: i32;
  var y: i32;
}

class Complex {
  fn Zero() -> Complex {
    return {.r = 0, .i = 0};
  }

  impl as Multipliable {
    fn Mul[me: Self](other: Self) -> Self {
       return {.r = me.r * other.r, .i = me.i * other.i};
    }
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
  return p.Square().x - c.Square().r + 7;
}
