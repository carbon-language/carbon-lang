// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: Class D
// CHECK:STDOUT: d.value = 2
// CHECK:STDOUT: d.GetValue() = 2
// CHECK:STDOUT: result: 0

package ExplorerTest api;

base class C {
  fn Method1() {
    Print("Class C");
  }
  fn GetValue[self: Self]() -> i32 {
    return self.value;
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
  var d: D = {.value = 2, .base={.value = 1}};
  d.Method1();

  Print("d.value = {0}", d.value);
  Print("d.GetValue() = {0}", d.GetValue());
  return 0;
}
