// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 1234

package ExplorerTest api;

interface A {
  fn F() -> i32;
}
interface B {
  fn F() -> i32;
  fn G() -> i32;
}
interface C(T:! Type) {
  fn H() -> T;
}

external impl i32 as A {
  fn F() -> i32 { return 1; }
}

external impl i32 as B & C(i32) where .Self is A {
  fn F() -> i32 { return 2; }
  fn G() -> i32 { return 3; }
  fn H() -> i32 { return 4; }
}

fn Main() -> i32 {
  let n: i32 = 0;
  return n.(A.F)() * 1000 + n.(B.F)() * 100 + n.(B.G)() * 10 + n.(C(i32).H)();
}
