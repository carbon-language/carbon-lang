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

class Point(T:! Type) {
  var x: T;
  var y: T;
}

fn Main() -> i32 {
  var p: Point(i32) = {.x = 0, .y = 0};
  // CHECK: COMPILATION ERROR: {{.*}}/executable_semantics/testdata/generic_class/fail_point_equal.carbon:[[@LINE+1]]: type error in name binding: 'class Point(T = i32)' is not implicitly convertible to 'class Point(T = Bool)'
  var q: Point(Bool) = p;
  return 0;
}
