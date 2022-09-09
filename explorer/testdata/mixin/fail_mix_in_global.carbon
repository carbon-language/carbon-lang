// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{not} %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s

package ExplorerTest api;

__mixin Operations {
  fn Square[me: Self](x:i32) -> i32{
    return x * x;
  }
}

// CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/mixin/fail-mix-in-global.carbon:[[@LINE+1]]: syntax error, unexpected MIX, expecting END_OF_FILE
__mix Operations;

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
  c.r = 3;
  return p.Square(p.y) - c.Square(c.i);
}
