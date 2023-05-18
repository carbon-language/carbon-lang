// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: 3
// CHECK:STDOUT: 7
// CHECK:STDOUT: 15
// CHECK:STDOUT: result: 0

package ExplorerTest api;

class A { var n: i32; }

external impl A as BitOrWith(i32) where .Result = A {
  fn Op[self: Self](rhs: i32) -> A { return {.n = self.n | rhs}; }
}

fn Main() -> i32 {
  var a: A = {.n = 1};
  a = a | 2;
  Print("{0}", a.n);
  a |= 5;
  Print("{0}", a.n);
  Print("{0}", (a | 12).n);
  return 0;
}
