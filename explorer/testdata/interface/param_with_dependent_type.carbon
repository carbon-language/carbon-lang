// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 42

package ExplorerTest api;

interface HasValueParam(T:! Type, V:! T) {
  fn Go[me: Self]() -> T;
}

impl () as HasValueParam(i32, 5) {
  fn Go[me: Self]() -> i32 { return 42; }
}

fn Main() -> i32 {
  return ().(HasValueParam(i32, 5).Go)();
}
