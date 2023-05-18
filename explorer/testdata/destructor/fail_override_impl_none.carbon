// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;


base class A {
    // CHECK:STDERR: COMPILATION ERROR: explorer/testdata/destructor/fail_override_impl_none.carbon:[[@LINE+1]]: Error declaring destructor for `A`: cannot override a destructor that is not declared `virtual` in base class.
    impl destructor[self: Self]{}
}

fn Main() -> i32 {
  return 0;
}
