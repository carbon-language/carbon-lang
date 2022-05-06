// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: RUNTIME ERROR: {{.*}}/explorer/testdata/member_access/qualified_param_member.carbon:23: could not find `impl binding T:! A as interface A`

package Foo api;
interface A {
  fn F[me: Self](o: Self) -> Self;
}
class X {
  impl as A {
    fn F[me: Self](o: Self) -> Self { return {.n = me.n + o.n}; }
  }
  var n: i32;
}
fn F[T:! A](v: T, w: T) -> T {
  return v.(T.F)(w);
}
fn Main() -> i32 {
  var v: X = {.n = 1};
  var w: X = {.n = 2};
  return F(v, w).n;
}
