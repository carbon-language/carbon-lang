// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | %{FileCheck-strict} %s
// RUN: %{not} %{explorer-trace} %s 2>&1 | %{FileCheck-allow-unmatched} %s
// AUTOUPDATE: %{explorer} %s

package ExplorerTest api;

// CHECK:COMPILATION ERROR: {{.*}}/explorer/testdata/function/fail_recurse_in_return_type.carbon:[[@LINE+1]]: 'T' is not usable until after it has been completely declared
fn T() -> T() { return i32; }

fn Main() -> i32 {
  return 0;
}
