// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test that the read from x triggers an error because x is dead.
// This test also demonstrates how by-reference free-variable capture
// is dangerous and can happen inside continuations.

fn capture() -> __Continuation {
  var x: Int = 1;
  __continuation k {
    var y: Int = x;
  }
  return k;
}

fn main() -> Int {
  var k: __Continuation = capture();
  __run k; // error, lifetime of x is over
  return 0;
}
