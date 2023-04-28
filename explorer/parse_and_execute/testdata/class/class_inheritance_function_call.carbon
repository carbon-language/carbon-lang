// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: Class C, call 0
// CHECK:STDOUT: Class D, call 0
// CHECK:STDOUT: Class C, call 1
// CHECK:STDOUT: Class C, call 2
// CHECK:STDOUT: Class D, call 2
// CHECK:STDOUT: result: 0

package ExplorerTest api;

base class C {
  fn FunctionC(var i: i32) {
    Print("Class C, call {0}", i);
  }
}

class D extends C {
  fn FunctionD(var i: i32) {
    C.FunctionC(i);
    Print("Class D, call {0}", i);
  }
}

fn Main() -> i32 {
  // Calling function from class type
  D.FunctionD(0);
  // Note: D.FunctionC(0); is not accessible

  // Calling function from class object
  var d: D = {.base={}};
  d.FunctionC(1);
  d.FunctionD(2);
  return 0;
}
