// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: Read var_c=1
// CHECK:STDOUT: Read var_d=2
// CHECK:STDOUT: Assign var_c=3
// CHECK:STDOUT: Assign var_d=4
// CHECK:STDOUT: result: 0

package ExplorerTest api;

base class C {
  var var_c: i32;
}

class D extends C {
  var var_d: i32;
}

fn Main() -> i32 {
  // Initialization
  var d: D = {.base = {.var_c= 1}, .var_d= 2};

  // Read
  Print("Read var_c={0}", d.var_c);
  Print("Read var_d={0}", d.var_d);

  // Assignment
  d.var_c = 3;
  d.var_d = 4;
  Print("Assign var_c={0}", d.var_c);
  Print("Assign var_d={0}", d.var_d);

  return 0;
}
