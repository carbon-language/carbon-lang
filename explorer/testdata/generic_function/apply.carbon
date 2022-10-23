// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: -2

package ExplorerTest api;

fn apply[T:! Type, U:! Type](f: __Fn (T) -> U, x: T) -> U {
  return f(x);
}

fn positive(x: bool) -> i32 {
  if (x) {
    return 2;
  } else {
    return -2;
  }
}

fn Main() -> i32 {
  return apply(positive, false);
}
