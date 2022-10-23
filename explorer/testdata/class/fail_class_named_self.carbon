// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

// Error: Can't use keyword `Self` as the name of a class.
// CHECK:STDERR: SYNTAX ERROR: {{.*}}/explorer/testdata/class/fail_class_named_self.carbon:[[@LINE+1]]: syntax error, unexpected SELF, expecting identifier
class Self {
  var x: i32;
  var y: i32;
}

fn Main() -> i32 {
  var p: Self = {.x = 1, .y = 2};
  return p.y - p.x - 1;
}
