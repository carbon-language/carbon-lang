// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{executable_semantics} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{not} %{executable_semantics} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{executable_semantics} %s
// CHECK: COMPILATION ERROR: {{.*}}/executable_semantics/testdata/interface/fail_no_impl.carbon:31: could not find implementation of interface Vector for class Point

package ExecutableSemanticsTest api;

interface Vector {
  fn Add[me: Self](b: Self) -> Self;
  fn Scale[me: Self](v: i32) -> Self;
}

class Point {
  var x: i32;
  var y: i32;
}

fn AddAndScaleGeneric[T:! Vector](a: T, b: T, s: i32) -> T {
  return a.Add(b).Scale(s);
}

fn Main() -> i32 {
  var a: Point = {.x = 0, .y = 0};
  var b: Point = {.x = 2, .y = 3};
  var p: Point = AddAndScaleGeneric(a, b, 3);
  return p.x - 6;
}
