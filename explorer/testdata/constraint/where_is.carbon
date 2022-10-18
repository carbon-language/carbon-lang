// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: i32.F
// CHECK:STDOUT: result: 0

package ExplorerTest api;

interface Base {
  fn F();
}

interface Extension {
  extends Base;
}

fn F[T:! Type where .Self is Extension](x: T) {
  x.(Extension.F)();
}

impl i32 as Extension {
  fn F() { Print("i32.F"); }
}

fn Main() -> i32 {
  var n: i32 = 0;
  F(n);
  return 0;
}
