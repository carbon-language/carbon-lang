// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package ExecutableSemanticsTest api;

// Assignment for continuations is shallow, so `k2` refers to the same
// continuation as `k1`.

fn main() -> i32 {
  var x: i32 = 0;
  __continuation k1 {
    x = x + 1;
    __await;
    x = x + 2;
  }
  var k2: __Continuation = k1;
  __run k1;
  __run k2;
  return x;
}
