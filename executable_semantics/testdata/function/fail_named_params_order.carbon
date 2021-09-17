// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: not executable_semantics %s 2>&1 | \
// RUN:   FileCheck --match-full-lines --allow-unused-prefixes=false %s
// RUN: not executable_semantics --trace %s 2>&1 | \
// RUN:   FileCheck --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: executable_semantics %s
// CHECK: PROGRAM ERROR: {{.*}}/executable_semantics/testdata/function/fail_named_params_order.carbon:14: positional members must come before named members

package ExecutableSemanticsTest api;

fn f(x: i32, .d = y: i32, z: i32, .e = a: i32) -> i32 {
  return (x + y) - (z + a);
}

fn main() -> i32 {
  return 0;
}
