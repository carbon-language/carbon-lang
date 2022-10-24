// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 0

package ExplorerTest api;

class C(T:! Type) {}

fn F(T:! Type) -> Type {
  return C(T);
}

fn Main() -> i32 {
  var v: F(i32) = {};
  var w: C(i32) = v;
  return 0;
}
