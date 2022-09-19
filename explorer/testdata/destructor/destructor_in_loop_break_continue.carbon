// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: DESTRUCTOR A 1
// CHECK: DESTRUCTOR A 1
// CHECK: DESTRUCTOR C 2
// CHECK: DESTRUCTOR B 3
// CHECK: DESTRUCTOR A 4
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

fn ident(x: bool)-> bool{
    return x;
}

fn ident_i32(x: i32)-> i32{
    return x;
}

fn Main() -> i32 {
  var i: i32 = 0;
  while(i < 2 ){
      var a: A = {.n = 4};
      var b: B = {.n = 3};
      var c: C = {.n = 2};
      if(i == 0){
        var a: A = {.n = 1};
        break;
      }
  }

  i = 0;
  while(i < 2 ){
    var a: A = {.n = 8};
    var b: B = {.n = 7};
    var c: C = {.n = 6};
    if(i == 0){
      var a: A = {.n = 5};
      i = i+1;
      continue;
    }
    i = i+1;
  }

  return 1;
}
