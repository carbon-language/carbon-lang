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

base class B {
  var a: A;
  var b: i32;
}

class C {
  extend base: B;
  var c: i32;
}

fn Main() -> i32 {
  var x: B = {.a = 1, .b = 2 as A};
  Print("1: {0}", x.a.n);
  Print("2: {0}", x.b);

  x = {.a = 3, .b = 4 as A};
  Print("3: {0}", x.a.n);
  Print("4: {0}", x.b);

  var y: C = {.base = {.a = 5, .b = 6 as A}, .c = 7 as A};
  Print("5: {0}", y.a.n);
  Print("6: {0}", y.b);
  Print("7: {0}", y.c);
  return 0;
}

// CHECK:STDOUT: 1: 1
// CHECK:STDOUT: 2: 2
// CHECK:STDOUT: 3: 3
// CHECK:STDOUT: 4: 4
// CHECK:STDOUT: 5: 5
// CHECK:STDOUT: 6: 6
// CHECK:STDOUT: 7: 7
// CHECK:STDOUT: result: 0
