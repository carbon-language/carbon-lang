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
  }

  var n: i32;
}

fn Main() -> i32 {
  var a: A = {.n = 2};
  return 0;
}

// CHECK:STDOUT: DESTRUCTOR A 3
// CHECK:STDOUT: result: 0
