// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

class A {
  fn F[self: Self]() -> type {
    // CHECK:STDERR: COMPILATION ERROR: explorer/testdata/class/fail_call_method_before_typecheck.carbon:[[@LINE+1]]: attempt to call function `F` that has not been fully type-checked
    var a: ({} as A).F() = 0;
    return i32;
  }
}

fn Main() -> i32 {
  return 0;
}
