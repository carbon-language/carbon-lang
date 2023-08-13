// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

class A {
  var n: i32;
  impl as ImplicitAs(i32) {
    fn Convert[self: Self]() -> i32 { return self.n; }
  }
}
impl i32 as ImplicitAs(A) {
  fn Convert[self: Self]() -> A { return {.n = self}; }
}

fn Main() -> i32 {
  var x: {.a: A, .b: i32} = {.a = 1, .b = 2 as A};
  Print("{0}", x.a.n);
  Print("{0}", x.b);

  var y: {.a: i32, .b: A} = {.a = 3, .b = 4 as A};
  x = y;
  Print("{0}", x.a.n);
  Print("{0}", x.b);
  return 0;
}

// CHECK:STDOUT: 1
// CHECK:STDOUT: 2
// CHECK:STDOUT: 3
// CHECK:STDOUT: 4
// CHECK:STDOUT: result: 0
