// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{executable_semantics} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{not} %{executable_semantics} --trace %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{executable_semantics} %s
// CHECK: COMPILATION ERROR: {{.*}}/executable_semantics/testdata/generic_class/fail_argument_deduction.carbon:26: type error in argument deduction

package ExecutableSemanticsTest api;

class Point(T:! Type) {
  var x: T;
  var y: T;
}

fn FirstOfTwoPoints[T:! Type](a: Point(T), b: Point(T)) -> Point(T) {
  return a;
}

fn Main() -> i32 {
  var p: Point(i32) = {.x = 0, .y = 1};
  var q: Point(Bool) = {.x = true, .y = false};
  return FirstOfTwoPoints(p, q).x;
}
