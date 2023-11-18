// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: B::Foo called!
// CHECK:STDOUT: result: 0

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

fn Main() -> i32 {
  var b: B = { .base = {} };
  b.Foo();
  return 0;
}
