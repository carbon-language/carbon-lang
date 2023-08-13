// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

class A {
  var a: i32;
}
class B {
  var b: i32;
}
class C {
  var c: i32;
}

impl A as ImplicitAs(B) {
  fn Convert[self: Self]() -> B { return {.b = self.a + 1}; }
}
impl B as As(C) {
  fn Convert[self: Self]() -> C { return {.c = self.b + 2}; }
}

fn Main() -> i32 {
  var a: (A, A) = ({.a = 1}, {.a = 2});
  var b: (B, B) = a;
  Print("{0}", b[0].b);
  Print("{0}", b[1].b);
  var c: (C, C) = b as (C, C);
  Print("{0}", c[0].c);
  Print("{0}", c[1].c);
  return 0;
}

// CHECK:STDOUT: 2
// CHECK:STDOUT: 3
// CHECK:STDOUT: 4
// CHECK:STDOUT: 5
// CHECK:STDOUT: result: 0
