// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: result: 12

package Foo api;
interface A {
  fn F[me: Self]() -> i32;
}
interface B {
  fn G(o: Self) -> i32;
}
alias C = A & B;
class X {
  impl as A {
    fn F[me: Self]() -> i32 { return 10 * me.n; }
  }
  impl as B {
    fn G(o: Self) -> i32 { return o.n; }
  }
  var n: i32;
}
fn Main() -> i32 {
  var v: X = {.n = 1};
  var w: X = {.n = 2};
  return v.(C.F)() + X.(C.G)(w);
}
