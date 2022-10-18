// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 0

package ExplorerTest api;

interface Frob {
  let Result:! Type;
  fn F[me: Self]() -> Result;
}

fn Use[T:! Frob](x: T) -> T.Result {
  var v: T.Result = x.F();
  return v;
}

impl i32 as Frob where .Result = i32 {
  fn F[me: Self]() -> i32 { return 0; }
}

fn Main() -> i32 {
  return Use(0);
}
