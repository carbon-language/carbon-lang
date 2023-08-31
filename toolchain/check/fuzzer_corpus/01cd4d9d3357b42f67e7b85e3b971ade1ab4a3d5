// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: result: 5

package ExplorerTest api;

class A {
  var n: i32;
  fn Get[self: Self]() -> i32 { return self.n; }
}

fn Main() -> i32 {
  return ({.n = 5} as A).Get();
}
