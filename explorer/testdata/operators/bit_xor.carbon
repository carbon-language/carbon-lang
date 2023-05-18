// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: 6
// CHECK:STDOUT: 0
// CHECK:STDOUT: -1
// CHECK:STDOUT: result: 0

package ExplorerTest api;

class A { var n: i32; }

external impl A as BitXorWith(i32) where .Result = A {
  fn Op[self: Self](rhs: i32) -> A { return {.n = self.n ^ rhs}; }
}

fn Main() -> i32 {
  var a: A = {.n = 5};
  a = a ^ 3;
  Print("{0}", a.n);
  a ^= 6;
  Print("{0}", a.n);
  Print("{0}", (a ^ -1).n);
  return 0;
}
