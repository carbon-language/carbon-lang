// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | %{FileCheck-strict} %s
// RUN: %{not} %{explorer-trace} %s 2>&1 | %{FileCheck-allow-unmatched} %s
// AUTOUPDATE: %{explorer} %s
package ExplorerTest api;

__mixin M1 {
  fn Square(x: i32) -> i32{
     return x * x;
  }
}

__mixin M2 {
  export fn Twice(x: i32) -> i32{
    return x + x;
  }
  __mix M1;
}

class C {
  __mix M2;
  fn Foo[me: Self](a: i32, b: i32) -> i32{
// CHECK:COMPILATION ERROR: {{.*}}/explorer/testdata/mixin/fail_use_non_export_member_from_class.carbon:[[@LINE+1]]: class C does not have a field named Square
     return me.Twice(3) + me.Square(5);
  }
}

fn Main() -> i32 {
  return 0;
}
