// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: not executable_semantics %s 2>&1 | \
// RUN:   FileCheck --match-full-lines --allow-unused-prefixes=false %s
// RUN: not executable_semantics --trace %s 2>&1 | \
// RUN:   FileCheck --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: executable_semantics %s
// CHECK: COMPILATION ERROR: {{.*}}/executable_semantics/testdata/name_lookup/fail_global_duplicate.carbon:15: Duplicate name `x` also found at {{.*}}/executable_semantics/testdata/name_lookup/fail_global_duplicate.carbon:14

package ExecutableSemanticsTest api;

var x: i32 = 0;
var x: i32 = 0;

fn main() -> i32 {
  return 0;
}
