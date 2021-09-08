// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package ExecutableSemanticsTest api;

// Test that the read from x triggers an error because x is dead.
// This test also demonstrates how by-reference free-variable capture
// is dangerous and can happen inside continuations.

fn capture() -> __Continuation {
  var x: i32 = 1;
  __continuation k {
    var y: i32 = x;
  }
  return k;
}

fn main() -> i32 {
  var k: __Continuation = capture();
  __run k; // error, lifetime of x is over
  return 0;
}
