// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

class One {
  extend impl as ImplicitAs(i32) {
    fn Convert[self: Self]() -> i32 { return 1; }
  }
}

class Two {
  extend impl as ImplicitAs(i32) {
    fn Convert[self: Self]() -> i32 { return 2; }
  }
}

class N {
  extend impl as ImplicitAs(i32) {
    fn Convert[self: Self]() -> i32 { return self.n; }
  }
  var n: i32;
}

fn Get(a: i32, b: i32, c: i32) -> i32 {
  return 100 * a + 10 * b + c;
}

fn Main() -> i32 {
  var i: One = {};
  var ii: Two = {};
  var iii: N = {.n = 3};
  return Get(i, ii, iii);
}

// CHECK:STDOUT: result: 123
