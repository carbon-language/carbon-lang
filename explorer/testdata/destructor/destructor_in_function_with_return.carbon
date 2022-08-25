// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: result: 7

package ExplorerTest api;



class A{
    fn destructor[me: Self]{
        Print("DESTRUCTOR A {0}",me.n);
    }
    var n: i32;
}

class B{
    fn destructor[me: Self]{
        Print("DESTRUCTOR B");
    }
}

class A0{
    fn destructor[me: Self]{
        Print("DESTRUCTOR A0");
    }
}

fn Main() -> i32 {
    var i : i32 = 0;
    if(i < 1){
        var a0: A = {.n = 3};
    }

    var a0: A = {.n = 2};
  return 1;
}
