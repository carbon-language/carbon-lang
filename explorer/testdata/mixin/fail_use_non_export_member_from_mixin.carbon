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
  export fn Cube[me:Self](x: i32) -> i32{
// CHECK:COMPILATION ERROR: {{.*}}/explorer/testdata/mixin/fail_use_non_export_member_from_mixin.carbon:[[@LINE+1]]: mixin M2 does not have a field named Square
    return x * me.Square(x);
  }
  __mix M1;
}

fn Main() -> i32 {
  return 0;
}
