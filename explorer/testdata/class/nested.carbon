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
  fn GetX[me: Point]() -> i32 {
    return me.x.Get();
  }

  class Coord {
    fn Get[me: Coord]() -> i32 {
      return me.p;
    }
    fn Zero() -> Coord {
      return {.p = 0};
    }

    var p: i32;
  }

  fn Origin() -> Point {
    // Should not fail, but does with error: could not resolve 'Coord'
    // return {.x = Coord.Zero(), .y = {.p = 0}};
    return {.x = Point.Coord.Zero(), .y = {.p = 0}};
  }

  // Should not fail, but does with error: could not resolve 'Coord'
  // var x: Coord;
  // Should not fail, but does with error: Coord is not a class function
  var x: Point.Coord;
  var y: Point.Coord;
}

fn Main() -> i32 {
  var p: Point = Point.Origin();
  var c: Point.Coord = Point.Coord.Zero();
  return p.GetX() + c.Get();
}
