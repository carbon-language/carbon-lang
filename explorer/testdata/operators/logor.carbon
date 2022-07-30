// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: result: true

package ExplorerTest api;

class A { var n: Bool; }

external impl A as LogOrWith(Bool) where .Result == A {
  fn Op[me: Self](rhs: Bool) -> A { return {.n = me.n or rhs}; }
}

fn Main() -> i32 {
  var a: A = {.n = true};
  a = a or false;
  return a.n;
}
