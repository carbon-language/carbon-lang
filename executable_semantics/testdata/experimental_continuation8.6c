// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test the way in which copying of continuations interacts with data
// on the stack such as the variable `x`. In this example the copy
// happens before the variable `x` is created, so each continuation
// creates a different `x`.

var Int: y = 0;

fn main() -> Int {
  __continuation k1 {
    var Int: x = 0;
    x = x + 1;
    __await;
    y = x;
  }
  var __Continuation: k2 = k1;
  __run k1;
  __run k2;
  __run k1;
  __run k2;
  return y;
}
