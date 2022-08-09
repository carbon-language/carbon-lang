// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: Read var_c=1
// CHECK: Read var_d=2
// CHECK: Assign var_c=3
// CHECK: Assign var_d=4
// CHECK: result: 0

package ExplorerTest api;

class C {
  var var_c: i32;
}

class D extends C {
  var var_d: i32;
}

fn Main() -> i32 {
  // Initialization
  var d: D = {.var_c= 1, .var_d= 2};

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
