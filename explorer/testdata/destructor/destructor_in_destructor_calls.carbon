// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: result: 1

package ExplorerTest api;

class A{
    fn destructor[me: Self]{
        Print("DESTRUCTOR A {0}",me.n);
    }
    fn Create(x: i32) -> A{
        return {.n = x};
    }
    var n: i32;
}

class B{
    fn destructor[me: Self]{
        Print("DESTRUCTOR B");
    }
    fn Create(x: A) -> B{
       return {.a = x};
    }
    var a: A;
}

class C{
    fn destructor[me: Self]{
        Print("DESTRUCTOR C");
    }
    fn Create(x: A, y : B) -> C{
       return {.a = x, .b = y};
    }
    var a: A;
    var b: B;
}


fn Main() -> i32 {
  var a: A = A.Create(2);
  var b: B = B.Create(a);
  var c: C = C.Create(a,b);
  return 1;
}
