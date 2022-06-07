// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: result: 52

package ExplorerTest api;

interface A { fn F() -> i32; fn G() -> i32; }
interface B { fn H() -> i32; }

fn Get1[T:! A & B](n: T) -> i32 { return n.F() + n.H(); }
fn Get2[T:! B & A](n: T) -> i32 { return n.G(); }

impl i32 as A {
  fn F() -> i32 { return 1; }
  fn G() -> i32 { return 2; }
}
impl i32 as B {
  fn H() -> i32 { return 4; }
}

fn Main() -> i32 {
  var z: i32 = 0;
  return Get1(z) * 10 + Get2(z);
}
