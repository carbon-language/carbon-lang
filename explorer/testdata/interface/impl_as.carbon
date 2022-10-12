// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | %{FileCheck-strict} %s
// RUN: %{explorer-trace} %s 2>&1 | %{FileCheck-allow-unmatched} %s
// AUTOUPDATE: %{explorer} %s
// CHECK:C.G
// CHECK:result: 5

package ExplorerTest api;

interface A(T:! Type) {
  fn F[me: Self]() -> T;
}

interface B {
  impl as A(i32);
  fn G[me: Self]();
}

class C {
  var n: i32;
}

external impl C as A(i32) {
  fn F[me: Self]() -> i32 { return me.n; }
}

external impl C as B {
  fn G[me: Self]() { Print("C.G"); }
}

fn H[T:! B](x: T) -> i32 {
  x.G();
  return x.(A(i32).F)();
}

fn Main() -> i32 {
  var c: C = {.n = 5};
  return H(c);
}
