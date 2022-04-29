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

class Point {
  var x: i32;
  var y: i32;
}

fn Main() -> i32 {
  var p: Point = {.x = 1, .y = 2};
  // CHECK: COMPILATION ERROR: {{.*}}/executable_semantics/testdata/class/fail_field_access_mismatch.carbon:[[@LINE+1]]: class Point does not have a field named z
  return p.z - 1;
}
