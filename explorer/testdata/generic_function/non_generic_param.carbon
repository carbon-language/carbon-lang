// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s | %{FileCheck-strict} %s
// RUN: %{explorer-trace} %s | %{FileCheck-allow-unmatched} %s
// AUTOUPDATE: %{explorer} %s
// CHECK:STDOUT: result: 1

package ExplorerTest api;

fn snd[T:! Type](x: i32, y: T) -> T {
  return y;
}

fn Main() -> i32 {
  return snd(0, 1);
}
