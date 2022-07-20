// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: result: 5

package ExplorerTest api;

class A { var n: i32; }

external impl A as ImplicitAs(i32) {
  fn Convert[me: Self]() -> i32 { return me.n; }
}

fn Main() -> i32 {
  var a: A = {.n = 5};
  return a as i32;
}
