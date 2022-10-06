// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | %{FileCheck-strict} %s
// RUN: %{explorer-trace} %s 2>&1 | %{FileCheck-allow-unmatched} %s
// AUTOUPDATE: %{explorer} %s
// CHECK:DESTRUCTOR A 1
// CHECK:DESTRUCTOR A 2
// CHECK:DESTRUCTOR A 3
// CHECK:DESTRUCTOR A 4
// CHECK:result: 2

package ExplorerTest api;

class A{
    destructor[me: Self]{
        Print("DESTRUCTOR A {0}",me.n);
    }
    var n: i32;
}

fn ident(x: bool)-> bool{
    return x;
}

fn ident_i32(x: i32)-> i32{
    return x;
}
// It should be enforced that different runtime scopes are on the stack.
//So that it can be tested that the wrong scopes are not removed from the stack.
fn Main() -> i32 {
  var i: i32 = 0;
  var d: A = {.n = 4};
  if(ident(true)){
    var a: A = {.n = 3};
    if(true){
       var b: A = {.n = 2};
       ident_i32(2);
       if(ident_i32(0) == 0){
         var c: A = {.n = 1};
         return 2;
       }
    }
  }
  return 1;
}
