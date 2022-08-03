// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{not} %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s

package ExplorerTest api;

fn fst[T:! Type](x: T, y: T) -> T {
  return x;
}

fn Main() -> i32 {
  // CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/generic_function/fail_type_deduction_mismatch.carbon:[[@LINE+3]]: deduced multiple different values for T:! Type:
  // CHECK:   i32
  // CHECK:   bool
  return fst(0, true);
}
