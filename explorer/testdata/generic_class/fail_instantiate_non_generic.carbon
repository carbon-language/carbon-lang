// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{not} %{explorer} --trace %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/generic_class/fail_instantiate_non_generic.carbon:23: attempt to instantiate a non-generic class: Point(i32)

package ExplorerTest api;

class Point {
  fn Origin() -> Point {
    return {.x = 0, .y = 0};
  }
  var x: i32;
  var y: i32;
}

fn Main() -> i32 {
  var p: Point(i32) = Point.Origin();
  return 0;
}
