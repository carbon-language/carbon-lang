// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: 4
// CHECK:STDOUT: 3
// CHECK:STDOUT: 2
// CHECK:STDOUT: result: 0

package ExplorerTest api;

class A { var n: i32; }

external impl A as SubWith(i32) where .Result = A {
  fn Op[self: Self](rhs: i32) -> A { return {.n = self.n - rhs}; }
}

fn Main() -> i32 {
  var a: A = {.n = 5};
  a = a - 1;
  Print("{0}", a.n);
  a -= 1;
  Print("{0}", a.n);
  Print("{0}", (a - 1).n);
  return 0;
}
