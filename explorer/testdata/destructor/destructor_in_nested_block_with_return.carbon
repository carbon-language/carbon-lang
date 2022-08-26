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
// CHECK: DESTRUCTOR A 4
// CHECK: result: 2

package ExplorerTest api;

class A{
    fn destructor[me: Self]{
        Print("DESTRUCTOR A {0}",me.n);
    }
    var n: i32;
}

class B{
    fn destructor[me: Self]{
        Print("DESTRUCTOR B {0}",me.n);
    }
    var n: i32;
}

class C{
    fn destructor[me: Self]{
        Print("DESTRUCTOR C {0}",me.n);
    }
    var n: i32;
}

fn ident(x: bool)-> bool{
    return x;
}

fn ident_i32(x: i32)-> i32{
    return x;
}

fn Main() -> i32 {
  var i: i32 = 0;
  var d: A = {.n = 4};
  if(ident(true)){
    var a: A = {.n = 3};
    if(i == 0){
       var b: B = {.n = 2};
       ident_i32(2);
       if(ident_i32(0) == 0){
         var c: C = {.n = 1};
         return 2;
       }
    }
  }
  return 1;
}
