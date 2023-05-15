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
  // CHECK:STDERR: COMPILATION ERROR: explorer/testdata/assert/fail_intrinsic_str_type.carbon:[[@LINE+3]]: type error in __intrinsic_assert argument 1
  // CHECK:STDERR: expected: String
  // CHECK:STDERR: actual: i32
  __intrinsic_assert(true, 1);
  return 0;
}
