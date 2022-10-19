// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | %{FileCheck-strict} %s
// RUN: %{explorer-trace} %s 2>&1 | %{FileCheck-allow-unmatched} %s
// AUTOUPDATE: %{explorer} %s
// CHECK:result: 1

package ExplorerTest api;

class A {
  var n: i32;
}
impl i32 as ImplicitAs(A) {
  fn Convert[me: Self]() -> A { return {.n = me }; }
}

fn Main() -> i32 {
  var a: A = {.n = 0};
  a = 1;
  return a.n;
}
