// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: A(T) as B(i32)
// CHECK:STDOUT: A(T) as B(U)
// CHECK:STDOUT: A(i32) as B(T)
// CHECK:STDOUT: A(i32) as B(T)
// CHECK:STDOUT: result: 0

package ExplorerTest api;

class A(T:! type) {}
interface B(T:! type) {
  fn F();
}

// Note: this match_first has no effect, because it only orders impls with the
// same type structure.
__match_first {
  external impl forall [T:! type] A(T) as B(i32) {
    fn F() {
      Print("A(T) as B(i32)");
    }
  }
  external impl forall [T:! type] A(i32) as B(T) {
    fn F() {
      Print("A(i32) as B(T)");
    }
  }
}

external impl forall [T:! type, U:! type] A(T) as B(U) {
  fn F() {
    Print("A(T) as B(U)");
  }
}

fn F[T:! B(i32)](x: T) {
  T.F();
}
fn G[T:! B(bool)](x: T) {
  T.F();
}

fn Main() -> i32 {
  let a: A(bool) = {};
  let b: A(i32) = {};
  F(a);
  G(a);
  F(b);
  G(b);
  return 0;
}
