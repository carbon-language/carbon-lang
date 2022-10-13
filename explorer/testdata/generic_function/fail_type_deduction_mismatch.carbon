// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s | %{FileCheck-strict} %s
// RUN: %{not} %{explorer-trace} %s | %{FileCheck-allow-unmatched} %s
// AUTOUPDATE: %{explorer} %s

package ExplorerTest api;

fn fst[T:! Type](x: T, y: T) -> T {
  return x;
}

fn Main() -> i32 {
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/generic_function/fail_type_deduction_mismatch.carbon:[[@LINE+3]]: deduced multiple different values for T:! Type:
  // CHECK:STDERR:   i32
  // CHECK:STDERR:   bool
  return fst(0, true);
}
