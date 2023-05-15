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
  // CHECK:STDERR: SYNTAX ERROR: explorer/testdata/assert/fail_intrinsic_bool_type.carbon:[[@LINE+1]]: syntax error, unexpected COMMA
  __intrinsic_assert(, "unused");
  return 0;
}
