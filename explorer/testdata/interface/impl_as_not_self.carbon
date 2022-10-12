// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | %{FileCheck-strict} %s
// RUN: %{explorer-trace} %s 2>&1 | %{FileCheck-allow-unmatched} %s
// AUTOUPDATE: %{explorer} %s
// CHECK:result: 5

package ExplorerTest api;

interface ConvertsFromInt {
  impl i32 as ImplicitAs(Self);
}

fn F(T:! ConvertsFromInt, n: i32) -> T {
  return n;
}

class C {
  var n: i32;
}
external impl i32 as ImplicitAs(C) {
  fn Convert[me: Self]() -> C { return {.n = me}; }
}
external impl C as ConvertsFromInt {}

fn Main() -> i32 {
  return F(C, 5).n;
}
