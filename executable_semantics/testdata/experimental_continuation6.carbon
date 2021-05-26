// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test recursive functions inside continuations.

var current: Int = 0;

fn CountUpTo(x: Int) -> Int {
  if (x == 0) {
    current = 0;
    __await;
    return 0;
  } else {
    current = 1 + CountUpTo(x - 1);
    __await;
    return current;
  }
}

fn main() -> Int {
  __continuation k {
    CountUpTo(5);
  }
  var sum: Int = 0;
  var count: Int = 5;
  while (not (count == 0)) {
    __run k;
    sum = sum + current;
    count = count - 1;
  }
  return sum;
}
