// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

class A {
  extend impl as ImplicitAs(i32) {
    fn Convert[self: Self]() -> i32 { return self.n; }
  }
  var n: i32;
}

fn Main() -> i32 {
  let a: A = {.n = 1};
  let b: A = {.n = 2};
  let (x: i32, var y: i32) = (a, b);
  return x + y;
}

// CHECK:STDOUT: result: 3
