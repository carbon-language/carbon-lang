// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: not executable_semantics %s 2>&1 | \
// RUN:   FileCheck --match-full-lines --allow-unused-prefixes=false %s
// RUN: not executable_semantics --trace %s 2>&1 | \
// RUN:   FileCheck --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: executable_semantics %s
// CHECK: COMPILATION ERROR: {{.*}}/executable_semantics/testdata/generic_function/fail_type_deduction_mismatch.carbon:21: type error in argument deduction
// CHECK: expected: i32
// CHECK: actual: Bool

package ExecutableSemanticsTest api;

fn fst[T:! Type](x: T, y: T) -> T {
  return x;
}

fn Main() -> i32 {
  return fst(0, true);
}
