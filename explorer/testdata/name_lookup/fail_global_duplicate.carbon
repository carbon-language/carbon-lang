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

var x: i32 = 0;
// CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/name_lookup/fail_global_duplicate.carbon:[[@LINE+1]]: Duplicate name `x` also found at {{.*}}/explorer/testdata/name_lookup/fail_global_duplicate.carbon:14
var x: i32 = 0;

fn Main() -> i32 {
  return 0;
}
