// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: Class D
// CHECK: d.value = 2
// CHECK: d.GetValue() = 2
// CHECK: result: 0

package ExplorerTest api;

base class C {
  fn Method1() {
    Print("Class C");
  }
  fn GetValue[me: Self]() -> i32 {
    return me.value;
  }
  var value: i32;
}

class D extends C {
  fn Method1() {
    Print("Class D");
  }
  var value: i32;
}

fn Main() -> i32 {
  // Initialize derived value first, base value second
  var d: D = {.value = 2, .value = 1};
  d.Method1();

  Print("d.value = {0}", d.value);
  Print("d.GetValue() = {0}", d.GetValue());
  return 0;
}
