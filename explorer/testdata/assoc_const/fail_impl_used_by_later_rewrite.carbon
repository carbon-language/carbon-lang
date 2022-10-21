// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

interface X(T:! Type) {}

interface Y(T:! Type) {
    let M:! X(T);
}

interface Z {
  // The `i32 is X(.Self)` constraint is indirectly required by
  // specifying that `.M = i32`.
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/assoc_const/fail_impl_used_by_later_rewrite.carbon:[[@LINE+1]]: could not find implementation of interface X(T = N) for i32
  let N:! Y(.Self) where i32 is X(.Self) and .M = i32;
}

impl i32 as X(i32) {}
impl i32 as Y(i32) where .M = i32 {}
impl i32 as Z where .N = i32 {}

fn F[A:! Z](a: A) -> A { return a; }

fn Main() -> i32 {
  return F(0);
}
