// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{not} %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/class/fail_class_named_self.carbon:14: syntax error, unexpected SELF, expecting identifier

package ExplorerTest api;

class Self {
  var x: i32;
  var y: i32;
}

fn Main() -> i32 {
  var p: Self = {.x = 1, .y = 2};
  return p.y - p.x - 1;
}
