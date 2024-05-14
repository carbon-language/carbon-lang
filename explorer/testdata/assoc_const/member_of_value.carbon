// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

interface Vector {
  let Dim:! i32;
}

class Point {
  var x: i32;
  var y: i32;
  extend impl as Vector where .Dim = 2 {}
}

fn Main() -> i32 {
  var a: Point = {.x = 2, .y = 1};
  return a.(Vector.Dim);
}

// CHECK:STDOUT: result: 2
