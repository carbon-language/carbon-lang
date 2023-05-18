// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

class ConvertTo(T:! type) {
  var v: T;
  impl as ImplicitAs(T) {
    fn Convert[self: Self]() -> T { return self.v; }
  }
}

fn Main() -> i32 {
  // CHECK:STDERR: COMPILATION ERROR: explorer/testdata/assert/fail_intrinsic_no_convert.carbon:[[@LINE+3]]: type error in __intrinsic_assert argument 0
  // CHECK:STDERR: expected: bool
  // CHECK:STDERR: actual: class ConvertTo(T = bool)
  __intrinsic_assert({.v = true} as ConvertTo(bool), {.v = "Pass"} as ConvertTo(String));
  return 0;
}
