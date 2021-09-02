// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package ExecutableSemanticsTest api;

// Test the way in which copying of continuations interacts with data
// on the stack such as the variable `x`. In this example the copy
// happens before the variable `x` is created, so each continuation
// creates a different `x`.

var y: i32 = 0;

fn main() -> i32 {
  __continuation k1 {
    var x: i32 = 0;
    x = x + 1;
    __await;
    y = x;
  }
  var k2: __Continuation = k1;
  __run k1;
  __run k2;
  __run k1;
  __run k2;
  return y;
}
