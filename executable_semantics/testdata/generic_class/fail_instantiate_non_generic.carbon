// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{executable_semantics} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{not} %{executable_semantics} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{executable_semantics} %s

package ExecutableSemanticsTest api;

class Point {
  fn Origin() -> Point {
    return {.x = 0, .y = 0};
  }
  var x: i32;
  var y: i32;
}

fn Main() -> i32 {
  // CHECK: COMPILATION ERROR: {{.*}}/executable_semantics/testdata/generic_class/fail_instantiate_non_generic.carbon:[[@LINE+1]]: attempt to instantiate a non-generic class: Point(i32)
  var p: Point(i32) = Point.Origin();
  return 0;
}
