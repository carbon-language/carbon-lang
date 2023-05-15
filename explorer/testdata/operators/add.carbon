// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: 12
// CHECK:STDOUT: 19
// CHECK:STDOUT: 26
// CHECK:STDOUT: result: 0

package ExplorerTest api;

class A { var n: i32; }

external impl A as Add {
  fn Op[self: Self](rhs: A) -> A { return {.n = self.n + rhs.n}; }
}

fn Main() -> i32 {
  var a: A = {.n = 5};
  var b: A = {.n = 7};
  a = a + b;
  Print("{0}", a.n);
  a += b;
  Print("{0}", a.n);
  Print("{0}", (a + b).n);
  return 0;
}
