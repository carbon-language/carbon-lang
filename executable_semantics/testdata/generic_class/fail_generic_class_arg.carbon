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
  fn Create(x: T, y: T) -> Point(T) {
    return {.x = x, .y = y};
  }

  var x: T;
  var y: T;
}

fn Main() -> i32 {
  // CHECK: COMPILATION ERROR: {{.*}}/executable_semantics/testdata/generic_class/fail_generic_class_arg.carbon:[[@LINE+1]]: type error in call: '(Type, Type)' is not implicitly convertible to '(Type)'
  var p: Point(i32) = Point(i32, i32).Create(0, 1);
  return p.x;
}
