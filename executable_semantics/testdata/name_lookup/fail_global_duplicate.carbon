// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{executable_semantics} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{not} %{executable_semantics} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{executable_semantics} %s

package ExecutableSemanticsTest api;

var x: i32 = 0;
// CHECK: COMPILATION ERROR: {{.*}}/executable_semantics/testdata/name_lookup/fail_global_duplicate.carbon:[[@LINE+1]]: Duplicate name `x` also found at {{.*}}/executable_semantics/testdata/name_lookup/fail_global_duplicate.carbon:13
var x: i32 = 0;

fn Main() -> i32 {
  return 0;
}
