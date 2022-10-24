// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 0

package ExplorerTest api;

interface Has(T:! Type) {
  fn Get() -> T;
}
impl i32 as Has(Type) {
  fn Get() -> Type { return Self; }
}

class WithType(T:! Has(Type)) {
  fn Get() -> Type { return T.Get(); }
}

fn Main() -> i32 {
  var v: WithType(i32).Get() = 0;
  return v;
}
