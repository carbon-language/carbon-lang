// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 5

package ExplorerTest api;

interface EqualConverter {
  let T:! Type;
  fn Convert(t: T) -> Self;
}

fn DoEqualConvert[T:! Type](t: T, U:! EqualConverter where .T = T) -> U {
  return U.Convert(t);
}

impl forall [U:! Type] U as EqualConverter where .T = U {
  fn Convert(u: U) -> U { return u; }
}

// Converts from T to U given only that T == U. Implemented without
// __intrinsic_as_equal_type, as proof that this is possible.
fn EqualConvert[T:! Type](t: T, U:! Type where .Self == T) -> U {
  return DoEqualConvert(t, U);
}

interface HasTwoTypes {
  let T:! Type;
  let U:! Type;
}

fn TestConvert(X:! HasTwoTypes where .T == .U, t: X.T) -> X.U {
  return EqualConvert(t, X.U);
}

impl i32 as HasTwoTypes where .T = i32 and .U = i32 {}

fn Main() -> i32 {
  return TestConvert(i32, 5);
}
