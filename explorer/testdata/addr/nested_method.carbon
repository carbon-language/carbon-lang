
// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 0

package ExplorerTest api;

class B {
  var x: i32;

  fn GetSetX[addr self: Self*](x: i32) -> i32 {
    var oldX: auto = (*self).x;
    (*self).x = x;
    return oldX;
  }
}

class A {
  var b: B;
}

fn Main() -> i32 {
  var b: B = {.x = 0};
  var a: A = {.b = b};
  var x: auto = a.b.GetSetX(42);
  if (a.b.x == 42) {
    return x;
  }
  return 1;
}
