// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest impl;

class A {
  destructor[addr self: Self*] {
    self->n += 1;
    Print("DESTRUCTOR A {0}", self->n);
  // CHECK:STDERR: RUNTIME ERROR: explorer/testdata/destructor/fail_addr.carbon:[[@LINE+1]]: destructors currently don't support `addr self` bindings
  }

  var n: i32;
}

fn Main() -> i32 {
  var a: A = {.n = 2};
  return 0;
}
