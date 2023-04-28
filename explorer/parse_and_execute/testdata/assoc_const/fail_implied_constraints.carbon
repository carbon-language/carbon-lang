// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

interface X(T:! type) {}

interface Y(T:! type) {
    let M:! X(T);
}

interface Z {
  // We reject this even though it is the responsibility of the `impl as Z` to
  // provide a type `N` such that `i32 is X(N)`. We might want to treat this as
  // an implied constraint and allow this in the future.
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/assoc_const/fail_implied_constraints.carbon:[[@LINE+1]]: could not find implementation of interface X(T = N) for i32
  let N:! Y(.Self) where .M = i32;
}

fn Main() -> i32 {
  return 0;
}
