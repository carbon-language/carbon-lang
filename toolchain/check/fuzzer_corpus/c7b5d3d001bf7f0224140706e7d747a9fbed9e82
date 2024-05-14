// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: Before: 1
// CHECK:STDOUT: Interface: 2
// CHECK:STDOUT: Op: 3
// CHECK:STDOUT: result: 0

package ExplorerTest api;

class C {
  var n: i32;
}

fn Main() -> i32 {
  var c: C = {.n = 1};
  Print("Before: {0}", c.n);
  c.(AssignWith({.n: i32}).Op)({.n = 2});
  Print("Interface: {0}", c.n);
  c = {.n = 3};
  Print("Op: {0}", c.n);
  return 0;
}
