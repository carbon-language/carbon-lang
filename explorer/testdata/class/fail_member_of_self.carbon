// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | %{FileCheck-strict} %s
// RUN: %{not} %{explorer-trace} %s 2>&1 | %{FileCheck-allow-unmatched} %s
// AUTOUPDATE: %{explorer} %s

package ExplorerTest api;

class C {
  // CHECK:COMPILATION ERROR: {{.*}}/explorer/testdata/class/fail_member_of_self.carbon:[[@LINE+1]]: incomplete type `class C` used in type of variable
  var m: Self;
}

fn Main() -> i32 {
  return 0;
}
