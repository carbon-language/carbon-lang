// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | %{FileCheck-strict} %s
// RUN: %{not} %{explorer-trace} %s 2>&1 | %{FileCheck-allow-unmatched} %s
// AUTOUPDATE: %{explorer} %s
package ExplorerTest api;


__mixin M {
  fn Square(x: i32) -> i32{
     return x * x;
  }
  export fn Twice(x: i32) -> i32{
    return x + x;
  }
}

class C {
  __mix M;
  fn Foo[me: Self](a: i32, b: i32) -> i32{
     var t: i32 = me.Twice(3);
// CHECK:COMPILATION ERROR: {{.*}}/explorer/testdata/mixin/fail_use_non_export_simple.carbon:[[@LINE+1]]: class C does not have a field named Square
     return t + me.Square(5);
  }
}

fn Main() -> i32 {
  return 0;
}
