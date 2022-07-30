// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: result: false

package ExplorerTest api;

class A { var n: Bool; }

external impl A as LogAndWith(Bool) where .Result == A {
  fn Op[me: Self](rhs: Bool) -> A { return {.n = me.n and rhs}; }
}

fn Main() -> i32 {
  var a: A = {.n = true};
  a = a and false;
  return a.n;
}
