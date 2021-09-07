// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: ../executable_semantics %s | \
// RUN:   FileCheck %s

package ExecutableSemanticsTest api;

fn swap[T:! Type, U:! Type](tuple: (T, U)) -> (U, T) {
  return (tuple[1], tuple[0]);
}

fn main() -> i32 {
  return swap((0, true))[1];
}

// AUTOUPDATE: ../executable_semantics %s
// CHECK: result: 0
