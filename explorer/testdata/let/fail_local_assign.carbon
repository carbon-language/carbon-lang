// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s | %{FileCheck-strict} %s
// RUN: %{not} %{explorer-trace} %s | %{FileCheck-allow-unmatched} %s
// AUTOUPDATE: %{explorer} %s

package ExplorerTest api;

fn Main() -> i32 {
  let x: auto = 10;
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/let/fail_local_assign.carbon:[[@LINE+1]]: Cannot assign to rvalue 'x'
  x = 0;
  return 0;
}
