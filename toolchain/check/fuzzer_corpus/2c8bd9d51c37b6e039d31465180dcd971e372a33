// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: result: 0

package ExplorerTest api;

fn ReturnIndirectly[T:! type](direct: bool, x: T) -> type {
  if (direct) {
    return T;
  } else {
    return ReturnIndirectly(true, x);
  }
}

fn Main() -> ReturnIndirectly(false, 0) {
  return 0;
}
