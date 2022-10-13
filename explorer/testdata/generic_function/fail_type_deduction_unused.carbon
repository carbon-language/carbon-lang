// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s | %{FileCheck-strict} %s
// RUN: %{not} %{explorer-trace} %s | %{FileCheck-allow-unmatched} %s
// AUTOUPDATE: %{explorer} %s

package ExplorerTest api;

fn id[T:! Type](x: i32) -> i32 {
  return x;
}

fn Main() -> i32 {
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/generic_function/fail_type_deduction_unused.carbon:[[@LINE+1]]: could not deduce type argument for type parameter T in call
  return id(0);
}
