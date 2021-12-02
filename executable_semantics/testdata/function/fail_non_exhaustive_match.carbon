// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: not executable_semantics %s 2>&1 | \
// RUN:   FileCheck --match-full-lines --allow-unused-prefixes=false %s
// RUN: not executable_semantics --trace %s 2>&1 | \
// RUN:   FileCheck --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: executable_semantics %s
// CHECK: COMPILATION ERROR: {{.*}}/executable_semantics/testdata/function/fail_non_exhaustive_match.carbon:18: non-exhaustive match may allow control-flow to reach the end of a function that provides a `->` return type

package ExecutableSemanticsTest api;

fn Main() -> i32 {
  match (0) {
    case 1 => return 0;
  }
}
