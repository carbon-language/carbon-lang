// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: 6
// CHECK:STDOUT: 5
// CHECK:STDOUT: 6
// CHECK:STDOUT: 5
// CHECK:STDOUT: 6
// CHECK:STDOUT: 5
// CHECK:STDOUT: 6
// CHECK:STDOUT: 5
// CHECK:STDOUT: result: 0

package ExplorerTest api;

class A { var n: i32; }

external impl A as Inc {
  fn Op[addr self: Self*]() { ++self->n; }
}
external impl A as Dec {
  fn Op[addr self: Self*]() { --self->n; }
}

fn Main() -> i32 {
  var a: A = {.n = 5};
  ++a.n;
  Print("{0}", a.n);
  --a.n;
  Print("{0}", a.n);
  ++a;
  Print("{0}", a.n);
  --a;
  Print("{0}", a.n);
  a.n.(Inc.Op)();
  Print("{0}", a.n);
  a.n.(Dec.Op)();
  Print("{0}", a.n);
  a.(Inc.Op)();
  Print("{0}", a.n);
  a.(Dec.Op)();
  Print("{0}", a.n);
  return 0;
}
