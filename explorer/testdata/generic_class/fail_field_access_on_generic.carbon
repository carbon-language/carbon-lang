// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | %{FileCheck-strict} %s
// RUN: %{not} %{explorer-trace} %s 2>&1 | %{FileCheck-allow-unmatched} %s
// AUTOUPDATE: %{explorer} %s

package ExplorerTest api;

fn BadSimpleMemberAccess[T:! Type](a: T) -> T {
  // CHECK:COMPILATION ERROR: {{.*}}/explorer/testdata/generic_class/fail_field_access_on_generic.carbon:[[@LINE+1]]: member access in unconstrained type
  return a.x;
}

fn Main() -> i32 {
  return BadSimpleMemberAccess(0);
}
