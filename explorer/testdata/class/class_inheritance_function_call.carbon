// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: Class C, call 0
// CHECK: Class D, call 0
// CHECK: Class C, call 1
// CHECK: Class C, call 2
// CHECK: Class D, call 2
// CHECK: result: 0

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
  var d: D = {};
  d.FunctionC(1);
  d.FunctionD(2);
  return 0;
}
