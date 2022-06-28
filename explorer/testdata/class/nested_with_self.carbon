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

class Point {
  // `Self` == `Point`
  fn GetX[me: Self]() -> i32 {
    return me.x.Get();
  }

  class Coord {
    // `Self` == `Coord`
    fn Get[me: Self]() -> i32 {
      return me.p;
    }
    fn Zero() -> Self {
      return {.p = 0};
    }

    var p: i32;
  }

  // `Self` == `Point`
  fn Origin() -> Self {
    return {.x = Coord.Zero(), .y = {.p = 0}};
  }

  var x: Coord;
  var y: Coord;
}

fn Main() -> i32 {
  var p: Point = Point.Origin();
  var c: Point.Coord = Point.Coord.Zero();
  return p.GetX() + c.Get();
}
