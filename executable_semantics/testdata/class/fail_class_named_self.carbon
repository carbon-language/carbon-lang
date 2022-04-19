// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{executable_semantics} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{not} %{executable_semantics} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{executable_semantics} %s
// CHECK: COMPILATION ERROR: {{.*}}/executable_semantics/testdata/class/fail_class_named_self.carbon:14: syntax error, unexpected SELF, expecting identifier

package ExecutableSemanticsTest api;

class Self {
  var x: i32;
  var y: i32;
}

fn Main() -> i32 {
  var p: Self = {.x = 1, .y = 2};
  return p.y - p.x - 1;
}
