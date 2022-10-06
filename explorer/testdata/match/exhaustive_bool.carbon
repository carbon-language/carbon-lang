// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | %{FileCheck-strict} %s
// RUN: %{explorer-trace} %s 2>&1 | %{FileCheck-allow-unmatched} %s
// AUTOUPDATE: %{explorer} %s
// CHECK:result: 2

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
