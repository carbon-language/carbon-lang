// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 526

package ExplorerTest api;

interface A { fn F() -> i32; fn G() -> i32; }
interface B { fn H() -> i32; }

fn Get1[T:! A & B](n: T) -> i32 { return n.F() + n.H(); }
fn Get2[T:! B & A](n: T) -> i32 { return n.G(); }
fn Get3[T:! B & A & A & B & A](n: T) -> i32 { return n.G() + n.H(); }

impl i32 as A {
  fn F() -> i32 { return 1; }
  fn G() -> i32 { return 2; }
}
impl i32 as B {
  fn H() -> i32 { return 4; }
}

fn Main() -> i32 {
  var z: i32 = 0;
  return Get1(z) * 100 + Get2(z) * 10 + Get3(z);
}
