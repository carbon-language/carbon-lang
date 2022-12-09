// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: DESTRUCTOR C 3
// CHECK:STDOUT: DESTRUCTOR A 1
// CHECK:STDOUT: result: 1

package ExplorerTest api;

base class A {
    destructor[self: Self]{
        Print("DESTRUCTOR A {0}", self.a);
    }
    var a: i32;
}

base class B extends A {
    var b: i32;
}

class C extends B {
    destructor[self: Self]{
        Print("DESTRUCTOR C {0}", self.c);
    }
    var c: i32;
}

fn Main() -> i32 {
  var c: C = { .base={ .base={ .a=1 }, .b=2}, .c=3 };
  return 1;
}
