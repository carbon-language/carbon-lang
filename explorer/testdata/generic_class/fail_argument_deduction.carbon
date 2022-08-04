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

class Point(T:! Type) {
  var x: T;
  var y: T;
}

fn FirstOfTwoPoints[T:! Type](a: Point(T), b: Point(T)) -> Point(T) {
  return a;
}

fn Main() -> i32 {
  var p: Point(i32) = {.x = 0, .y = 1};
  var q: Point(bool) = {.x = true, .y = false};
  // CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/generic_class/fail_argument_deduction.carbon:[[@LINE+3]]: deduced multiple different values for T:! Type:
  // CHECK:   i32
  // CHECK:   bool
  return FirstOfTwoPoints(p, q).x;
}
