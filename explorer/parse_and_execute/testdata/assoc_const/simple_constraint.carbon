// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 3

package ExplorerTest api;

interface Frob {
  let Result:! type;
  fn F[self: Self]() -> Result;
}

fn Use[T:! Frob where .Result = .Self](x: T) -> T {
  var v: T = x.F();
  return v;
}

impl i32 as Frob where .Result = i32 {
  fn F[self: Self]() -> i32 { return self + 1; }
}

fn Main() -> i32 {
  return Use(2);
}
