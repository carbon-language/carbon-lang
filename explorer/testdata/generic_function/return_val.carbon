// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

fn id[T:! type](x: T) -> T {
  return x;
}

fn Main() -> i32 {
  return id(0);
}

// CHECK:STDOUT: result: 0
