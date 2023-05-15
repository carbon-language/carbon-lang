// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;


base class A {
    virtual destructor[self: Self]{}
}

class B extends A {
    // CHECK:STDERR: COMPILATION ERROR: explorer/testdata/destructor/fail_override_virtual_virtual.carbon:[[@LINE+1]]: Error declaring destructor for `B`: use `impl` to implement virtual destructor in child class.
    virtual destructor[self: Self]{}
}

fn Main() -> i32 {
  return 0;
}
