// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 0

package ExplorerTest api;

interface Similar(T:! Type) {}

impl forall [T:! Type] T as Similar(T) {}
impl forall [T:! Type] i32 as Similar(T) {}

fn CheckSimilar[T:! Type, U:! Similar(T)](a: T, b: U) {}

fn Main() -> i32 {
  let n: i32 = 0;
  CheckSimilar(true, false);
  CheckSimilar(true, n);
  return 0;
}
