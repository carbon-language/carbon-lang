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
  export fn Cube[me: Self](x:i32) -> i32{
  //CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/mixin/fail_reuse_in_declaration.carbon:[[@LINE+1]]: mixin Operations does not have a field named Square
    return x * me.Square(x);
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

fn Main() -> i32 {
  var p: Point = Point.Origin();
  p.x = 3;
  return p.Cube(p.x) - 27;
}
