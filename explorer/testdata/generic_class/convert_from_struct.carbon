// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: result: 5

package ExplorerTest api;

class Point(T:! Type) {
  var x: T;
  var y: T;
}

fn GetX[T:! Type](pt: Point(T)) -> T {
  return pt.x;
}
fn GetY(T:! Type, pt: Point(T)) -> T {
  return pt.y;
}

fn Main() -> i32 {
  var p: Point(i32) = {.x = 1, .y = 2};
  // FIXME: Should `GetX({.x = 1, .y = 2})` work? See #1251.
  return GetX(p) + GetY(i32, {.x = 3, .y = 4});
}
