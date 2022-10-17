// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: Class 1
// CHECK: Class 2
// CHECK: result: 0

package ExplorerTest api;

base class C {
  fn BasePrint(v: i32) {
    Print("Class {0}", v);
  }
  fn Method1[me:Self]() {
    me.BasePrint(me.value_c);
  }

  var value_c: i32;
}

class D extends C {
  fn Method2[me:Self]() {
    me.BasePrint(me.value_d);
  }

  var value_d: i32;
}

fn Main() -> i32 {
  var d: D = {.value_c = 1, .value_d = 2};
  d.Method1();
  d.Method2();
  return 0;
}
