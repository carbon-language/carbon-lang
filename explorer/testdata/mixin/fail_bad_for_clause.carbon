// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | %{FileCheck-strict} %s
// RUN: %{not} %{explorer-trace} %s 2>&1 | %{FileCheck-allow-unmatched} %s
// AUTOUPDATE: %{explorer} %s

package ExplorerTest api;
__mixin M for i32 {
  export fn Square[me: Self](x:i32) -> i32 {
    return x * x;
  }
// CHECK:COMPILATION ERROR: {{.*}}/explorer/testdata/mixin/fail_bad_for_clause.carbon:[[@LINE+1]]: Expected constraint after `for`, found i32
}

fn Main() -> i32 {
  return 0;
}
