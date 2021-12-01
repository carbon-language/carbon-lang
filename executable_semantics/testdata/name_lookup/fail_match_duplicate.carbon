// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: not executable_semantics %s 2>&1 | \
// RUN:   FileCheck --match-full-lines --allow-unused-prefixes=false %s
// RUN: not executable_semantics --trace %s 2>&1 | \
// RUN:   FileCheck --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: executable_semantics %s
// CHECK: COMPILATION ERROR: {{.*}}/executable_semantics/testdata/name_lookup/fail_match_duplicate.carbon:17: Duplicate name `x` also found at {{.*}}/executable_semantics/testdata/name_lookup/fail_match_duplicate.carbon:16

package ExecutableSemanticsTest api;

fn Main() -> i32 {
  match (0) {
    case x: i32 =>
      var x: i32 = 0;
  }
  return 0;
}
