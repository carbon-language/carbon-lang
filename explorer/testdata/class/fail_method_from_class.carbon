// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{not} %{explorer} --trace %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/class/fail_method_from_class.carbon:29: GetX is not a class function

package ExplorerTest api;

class Point {

  fn Origin() -> Point {
    return {.x = 0, .y = 0};
  }

  fn GetX[me: Point]() -> i32 {
    return me.x;
  }

  var x: i32;
  var y: i32;
}

fn Main() -> i32 {
  return Point.GetX();
}
