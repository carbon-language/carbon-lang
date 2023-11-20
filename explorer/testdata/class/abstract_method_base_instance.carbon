// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}

package ExplorerTest api;

abstract class A {
    abstract fn Foo[self: Self]();
}

base class B {
    extend base: A;
    impl fn Foo[self: Self]() {
      Print("B::Foo called!");
    }
}

class C {
    extend base: A;
    impl fn Foo[self: Self]() {
      Print("C::Foo called!");
    }
}

abstract class D {
    extend base: A;
}

class E {
    extend base: D;
    impl fn Foo[self: Self]() {
      Print("E::Foo called!");
    }
}


fn Main() -> i32 {
  var b: B = { .base = {} };
  b.Foo();
  var c: C = { .base = {} };
  c.Foo();
  var e: E = { .base = { .base = {} } };
  e.Foo();
  return 0;
}

// CHECK:STDOUT: B::Foo called!
// CHECK:STDOUT: C::Foo called!
// CHECK:STDOUT: E::Foo called!
// CHECK:STDOUT: result: 0
