// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{executable_semantics} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{executable_semantics} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{executable_semantics} %s
// CHECK: result: 0

package ExecutableSemanticsTest api;

interface Number {
  fn Zero() -> Self;
  fn Add[me: Self](other: Self) -> Self;
}

class Point(T:! Number) {
  fn Origin() -> Point(T) {
    return {.x = T.Zero(), .y = T.Zero()};
  }
  fn Clone[me: Point(T)]() -> Point(T) {
    return {.x = me.x, .y = me.y};
  }
  fn SumXY[me: Point(T)]() -> T {
    return me.x.Add(me.y);
  }
  var x: T;
  var y: T;
}

external impl i32 as Number {
  fn Zero() -> i32 { return 0; }
  fn Add[me: i32](other: i32) -> i32 { return me + other; }
}

fn Main() -> i32 {
  var p: Point(i32) = Point(i32).Origin();
  return p.Clone().SumXY();
}
