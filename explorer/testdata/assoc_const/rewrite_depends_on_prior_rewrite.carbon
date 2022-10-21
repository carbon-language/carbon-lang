// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 5

package ExplorerTest api;

interface HasTypeAndValue {
  let T:! Type;
  let V:! T;
}

fn F(X:! HasTypeAndValue where .T = i32 and .V = 5) -> i32 { return X.V; }

impl i32 as HasTypeAndValue where .T = i32 and .V = 5 {}

fn Main() -> i32 {
  return F(i32);
}
