// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{not} %{explorer} --trace %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/name_lookup/fail_block_duplicate.carbon:16: Duplicate name `x` also found at {{.*}}/explorer/testdata/name_lookup/fail_block_duplicate.carbon:15

package ExplorerTest api;

fn Main() -> i32 {
  var x: i32 = 0;
  var x: i32 = 0;
  return 0;
}
