// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 1

package ExplorerTest api;

interface Container {
  let Element:! Type;
  fn Front[me: Self]() -> Element;
}

fn A[T:! Container where .Element = i32](x: T) -> T.Element {
  return x.Front();
}

fn B[T:! Container where .Element == i32](x: T) -> T.Element {
  return A(x);
}

external impl (i32, i32) as Container where .Element = i32 {
  fn Front[me: Self]() -> i32 {
    let (a: i32, b: i32) = me;
    return a;
  }
}

fn Main() -> i32 {
  return B((1, 2));
}
