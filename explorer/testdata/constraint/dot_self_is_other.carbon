// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 12

package ExplorerTest api;

interface A { fn F() -> i32; }
interface B { fn F() -> i32; }

impl i32 as A { fn F() -> i32 { return 1; } }
impl i32 as B { fn F() -> i32 { return 2; } }

fn GetA[T:! A](x: T) -> i32 { return x.F(); }
fn GetAB[T:! B where .Self is A](x: T) -> i32 {
  // Note that there's no ambiguity in the use of the name F here.
  // `x.F` is `x.(B.F)` because the constraint only treats `B`, not
  // `A`, as a lookup context.
  return 10 * GetA(x) + x.F();
}

fn Main() -> i32 {
  var v: i32 = 0;
  return GetAB(v);
}
