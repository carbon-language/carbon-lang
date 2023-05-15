// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

fn fst[T:! type](x: T, y: T) -> T {
  return x;
}

fn Main() -> i32 {
  // CHECK:STDERR: COMPILATION ERROR: explorer/testdata/generic_function/fail_type_deduction_mismatch.carbon:[[@LINE+3]]: deduced multiple different values for T:! type:
  // CHECK:STDERR:   i32
  // CHECK:STDERR:   bool
  return fst(0, true);
}
