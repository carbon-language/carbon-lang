// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

class IntLike {
  var n: i32;
}

impl i32 as ImplicitAs(IntLike) {
  fn Convert[self: i32]() -> IntLike { return {.n = self}; }
}

fn Main() -> i32 {
  var arr: [i32; 4] = (0, 1, 2, 3);
  // TODO: We should accept this, but currently have nowhere in our
  // representation to describe the conversion.
  for (x: IntLike in arr) {
    Print("{0}", x.n);
  // CHECK:STDERR: COMPILATION ERROR: explorer/testdata/for/fail_convert.carbon:[[@LINE+3]]: type error in `for` pattern
  // CHECK:STDERR: expected: i32
  // CHECK:STDERR: actual: class IntLike
  }
  return 0;
}
