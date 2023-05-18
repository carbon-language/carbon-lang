// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: Destructor A 3
// CHECK:STDOUT: Destructor A 1
// CHECK:STDOUT: result: 0

package ExplorerTest api;

class A {
  var i: i32;
  destructor[self: Self] {
    Print("Destructor A {0}", self.i);
  }
}

fn Main() -> i32 {
  var a0: A;
  var a1: A = {.i = 1};
  var a2: A;
  var a3: A = {.i = 3};
  return 0;
}
