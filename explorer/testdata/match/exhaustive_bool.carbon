// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 2

package ExplorerTest api;

fn A(b: bool) -> i32 {
  match (b) {
    case false => { return 1; }
    case true => { return 2; }
  }
}

fn Main() -> i32 {
  return A(true);
}
