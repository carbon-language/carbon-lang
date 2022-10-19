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
// CHECK:DESTRUCTOR A 5
// CHECK:DESTRUCTOR A 6
// CHECK:DESTRUCTOR A 7
// CHECK:DESTRUCTOR A 8
// CHECK:DESTRUCTOR A 6
// CHECK:DESTRUCTOR A 7
// CHECK:DESTRUCTOR A 8
// CHECK:result: 1

package ExplorerTest api;

class A{
    destructor[me: Self]{
        Print("DESTRUCTOR A {0}",me.n);
    }
    var n: i32;
}

fn Main() -> i32 {
  var i: i32 = 0;
  while(true){
      var a: A = {.n = 4};
      var b: A = {.n = 3};
      var c: A = {.n = 2};
      if(true){
        var a: A = {.n = 1};
        break;
      }
  }

  i = 0;
  while(i < 2 ){
    var a: A = {.n = 8};
    var b: A = {.n = 7};
    var c: A = {.n = 6};
    if(i == 0){
      var a: A = {.n = 5};
      i = i+1;
      continue;
    }
    i = i+1;
  }

  return 1;
}
