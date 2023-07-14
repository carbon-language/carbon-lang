// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

class ConvertTo(T:! type) {
  var v: T;
  extend impl as ImplicitAs(T) {
    fn Convert[self: Self]() -> T { return self.v; }
  }
}

fn Main() -> i32 {
  // CHECK:STDERR: COMPILATION ERROR: fail_intrinsic_no_args.carbon:[[@LINE+1]]: __intrinsic_assert takes 2 arguments
  __intrinsic_assert();
  return 0;
}
