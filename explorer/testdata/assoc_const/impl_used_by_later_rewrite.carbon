// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 0

package ExplorerTest api;

interface X(T:! Type) {}

interface Y(T:! Type) {
    let M:! X(T);
}

interface Z {
  // The `i32 is X(.Self)` constraint is indirectly required by
  // specifying that `.M = i32`.
  let N:! Y(.Self) where i32 is X(.Self) and .M = i32;
}

impl i32 as X(i32) {}
impl i32 as Y(i32) where .M = i32 {}
impl i32 as Z where .N = i32 {}

fn F[A:! Z](a: A) -> A { return a; }

fn Main() -> i32 {
  return F(0);
}
