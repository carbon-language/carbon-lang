// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

fn Main() -> i32 {
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/struct/fail_struct_type_literal_duplicate_member.carbon:[[@LINE+1]]: Duplicate name `x` in struct type literal
  var y: {.x: i32, .x: i32};
  return 0;
}
