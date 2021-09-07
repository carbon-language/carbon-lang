// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: ../executable_semantics %s | \
// RUN:   FileCheck %s

package ExecutableSemanticsTest api;

fn snd[T:! Type](x: i32, y: T) -> T {
  return y;
}

fn main() -> i32 {
  return snd(0, 1);
}

// AUTOUPDATE: ../executable_semantics %s
// CHECK: result: 1
