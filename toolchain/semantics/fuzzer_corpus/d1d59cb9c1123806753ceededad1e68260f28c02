// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: result: 4

package ExplorerTest api;

fn F[T:! ImplicitAs(type)](x: T) -> type { return x; }

fn Main() -> i32 {
  var v: (i32, i32) as type = (1, 2);
  var w: F((i32, i32)) = (3, 4);
  v = w;
  return v[1];
}
