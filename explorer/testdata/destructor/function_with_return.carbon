// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: DESTRUCTOR C 1
// CHECK:STDOUT: DESTRUCTOR B 2
// CHECK:STDOUT: DESTRUCTOR A 3
// CHECK:STDOUT: result: 1

package ExplorerTest api;

class A{
    destructor[self: Self]{
        Print("DESTRUCTOR A {0}",self.n);
    }
    var n: i32;
}

class B{
    destructor[self: Self]{
        Print("DESTRUCTOR B {0}",self.n);
    }
    var n: i32;
}

class C{
    destructor[self: Self]{
        Print("DESTRUCTOR C {0}",self.n);
    }
    var n: i32;
}

fn Main() -> i32 {
  var a: A = {.n = 3};
  var b: B = {.n = 2};
  var c: C = {.n = 1};
  return 1;
}
