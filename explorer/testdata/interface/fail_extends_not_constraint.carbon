// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | %{FileCheck-strict} %s
// RUN: %{not} %{explorer-trace} %s 2>&1 | %{FileCheck-allow-unmatched} %s
// AUTOUPDATE: %{explorer} %s

package ExplorerTest api;

interface Vector {
  // CHECK:COMPILATION ERROR: {{.*}}/explorer/testdata/interface/fail_extends_not_constraint.carbon:[[@LINE+1]]: an interface can only extend a constraint, found i32
  extends i32;
}

fn Main() -> i32 {
  return 0;
}
