// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: result: 3

package ExplorerTest api;

class A {
  impl as ImplicitAs(i32) {
    fn Convert[me: Self]() -> i32 { return me.n; }
  }
  var n: i32;
}

fn Main() -> i32 {
  let a: A = {.n = 1};
  let b: A = {.n = 2};
  let (x: i32, var y: i32) = (a, b);
  return x + y;
}
