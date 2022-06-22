// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: result: 1232

package ExplorerTest api;

interface A {
  fn F() -> i32;
}
interface B {
  fn F() -> i32;
  fn G() -> i32;
}
interface C(T:! Type) {
  fn F() -> T;
}

external impl i32 as A {
  fn F() -> i32 { return 1; }
}

// TODO: Use `where .Self is A` once `.Self` support lands.
external impl i32 as B & C(i32) where i32 is A {
  fn F() -> i32 { return 2; }
  fn G() -> i32 { return 3; }
}

fn Main() -> i32 {
  let n: i32 = 0;
  return n.(A.F)() * 1000 + n.(B.F)() * 100 + n.(B.G)() * 10 + n.(C(i32).F)();
}
