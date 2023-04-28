// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 1

package ExplorerTest api;

interface Frob {
  let Result:! type;
  fn F[self: Self]() -> Result;
}

fn Use[T:! Frob](x: T) -> T.Result {
  var v: T.Result = x.F();
  return v;
}

class AlmostI32 {
  var val: i32;
  impl as ImplicitAs(i32) {
    fn Convert[self: Self]() -> i32 { return self.val; }
  }
}

impl i32 as Frob where .Result = AlmostI32 {
  fn F[self: Self]() -> AlmostI32 { return {.val = self}; }
}

fn Main() -> i32 {
  // Ensure that lookup for
  //   i32.(Frob.Result) as ImplicitAs(i32)
  // finds `impl AlmostI32 as ImplicitAs(i32)`.
  return Use(1);
}
