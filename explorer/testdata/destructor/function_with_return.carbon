// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: DESTRUCTOR C 1
// CHECK: DESTRUCTOR B 2
// CHECK: DESTRUCTOR A 3
// CHECK: result: 1

package ExplorerTest api;

class A{
    destructor[me: Self]{
        Print("DESTRUCTOR A {0}",me.n);
    }
    var n: i32;
}

class B{
    destructor[me: Self]{
        Print("DESTRUCTOR B {0}",me.n);
    }
    var n: i32;
}

class C{
    destructor[me: Self]{
        Print("DESTRUCTOR C {0}",me.n);
    }
    var n: i32;
}

fn Main() -> i32 {
  var a: A = {.n = 3};
  var b: B = {.n = 2};
  var c: C = {.n = 1};
  return 1;
}
