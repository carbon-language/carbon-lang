// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: Destroying 0 (C)
// CHECK:STDOUT: Destroying 1
// CHECK:STDOUT: Destroying 2
// CHECK:STDOUT: Destroying 3 (B)
// CHECK:STDOUT: Destroying 4
// CHECK:STDOUT: Destroying 5
// CHECK:STDOUT: Destroying 6 (A)
// CHECK:STDOUT: Destroying 7
// CHECK:STDOUT: Destroying 8
// CHECK:STDOUT: result: 0

package ExplorerTest api;

base class Data {
  fn Make(x: i32) -> Data { return {.data = x}; }
  destructor[self: Self]{
    Print("Destroying {0}", self.data);
  }
  var data: i32;
}

base class A {
  fn Make() -> A { return {.a1=Data.Make(8), .a2=Data.Make(7)}; }
  destructor[self: Self]{
    Print("Destroying 6 (A)");
  }
  var a1: Data;
  var a2: Data;
}

base class B extends A {
  fn Make() -> B { return {.base = A.Make(), .b1=Data.Make(5), .b2=Data.Make(4)}; }
  destructor[self: Self]{
    Print("Destroying 3 (B)");
  }
  var b1: Data;
  var b2: Data;
}

class C extends B {
  fn Make() -> C { return {.base = B.Make(), .c1=Data.Make(2), .c2=Data.Make(1)}; }
  destructor[self: Self]{
    Print("Destroying 0 (C)");
  }
  var c1: Data;
  var c2: Data;
}

fn Main() -> i32 {
  var c: C = C.Make();
  return 0;
}
