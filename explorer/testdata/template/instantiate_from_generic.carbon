// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

interface I { fn F[self: Self](); }

fn CheckTimplsI[T:! I](x: T) { x.F(); }

interface J { fn F[self: Self](); }

impl forall [template T:! I] T as J {
  fn F[self: Self]() { CheckTimplsI(self); }
}

// Ensure that the instantiated `impl T as J` is type-checked in an impl scope
// where `T impls I` is available, so that its call to `CheckTimplsI` is valid.
fn UseTemplatedImplFromGeneric[T:! I](x: T) { x.(J.F)(); }

impl i32 as I {
  fn F[self: i32]() {
    Print("{0}", self);
  }
}

fn Main() -> i32 {
  UseTemplatedImplFromGeneric(1);
  return 0;
}

// CHECK:STDOUT: 1
// CHECK:STDOUT: result: 0
