// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;
interface Multipliable {
  fn Mul[me: Self](other: Self) -> Self;
}

__mixin Operations for Multipliable {
  export fn Square[me: Self]() -> Self {
    return me.Mul(me);
  }
}

__mixin M {
  export __mix Operations;
  // CHECK:STDERR: ...
}

class Point {
  fn Origin() -> Point {
    return {.x = 0, .y = 0};
  }

  __mix M;

  impl as Multipliable {
    fn Mul[me: Self](other: Self) -> Self {
       return {.x = me.x * other.x, .y = me.y * other.y};
    }
  }

  var x: i32;
  var y: i32;
}

fn Main() -> i32 {
   var p: Point = Point.Origin();
   p.x = 3;
   return p.Square().x - 9;
}
