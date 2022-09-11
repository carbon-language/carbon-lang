// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: result: 0

package ExplorerTest api;

__mixin Operations {
  fn Square[me: Self](x:i32) -> i32{
    return x * x;
  }
}

class Point {
  var x: i32;
  var y: i32;
  fn DistanceSquare[me:Self](other:Self) -> i32 {
    return me.Square(me.x - other.x) + me.Square(me.y - other.y);
  }
  __mix Operations;
}

class Complex {
  var r: i32;
  var i: i32;
  __mix Operations;
  fn AbsSquare[me:Self]() -> i32 {
    return me.Square(me.r) + me.Square(me.i);
  }
}


fn Main() -> i32 {
  var p1: Point = {.x = 1, .y = 2 };
  var p2: Point = {.x = 4, .y = 3 };
  var c: Complex = {.r = 5, .i = 6 };
  return c.AbsSquare() - p1.DistanceSquare(p2) - 51;
}
