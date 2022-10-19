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
  fn Factorial[me: Self](x:i32) -> i32{
    if (x < 2) {
       return 1;
    }
    return x * me.Factorial(x - 1);
  }
  fn F[me: Self](x:i32, y:i32) -> i32{
    return me.Factorial(x) - me.Factorial(y);
  }
}

class Point {
  fn Origin() -> Point {
    return {.x = 0, .y = 0};
  }

  var x: i32;
  var y: i32;

  fn Foo[me:Self]() -> i32 {
     return me.F(me.x, me.y);
  }
  __mix Operations;
}

fn Main() -> i32 {
  var p: Point = Point.Origin();
  p.x = 5;
  p.y = 3;
  return p.Foo() - 114;
}
