// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: 1
// CHECK:STDOUT: result: 0

package ExplorerTest api;

interface I { fn F[self: Self](); }

fn I_F[T:! I](x: T) { x.F(); }

interface J { fn F[self: Self](); }

impl forall [template T:! I] T as J {
  fn F[self: Self]() { I_F(self); }
}

fn J_F[T:! I](x: T) { x.(J.F)(); }

impl i32 as I {
  fn F[self: i32]() {
    Print("{0}", self);
  }
}

fn Main() -> i32 {
  J_F(1);
  return 0;
}
