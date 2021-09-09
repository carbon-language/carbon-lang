// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: not executable_semantics %s 2>&1 | \
// RUN:   FileCheck --match-full-lines --allow-unused-prefixes=false %s
// RUN: not executable_semantics --trace %s 2>&1 | \
// RUN:   FileCheck --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: executable_semantics %s
// CHECK: COMPILATION ERROR: {{.*}}/generic_function_fail3.carbon:14: type error in addition(1)
// CHECK: expected: i32
// CHECK: actual: T


package ExecutableSemanticsTest api;

fn id[T:! Type](x: T) -> T {
  return x + 0;
}

fn main() -> i32 {
  return id(0);
}
