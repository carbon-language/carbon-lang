// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Assignment for continuations is shallow, so `k2` refers to the same
// continuation as `k1`.

fn main() -> Int {
  var Int: x = 0;
  __continuation k1 {
    x = x + 1;
    __await;
    x = x + 2;
  }
  var __Continuation: k2 = k1;
  __run k1;
  __run k2;
  return x;
}
