// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{not} %{explorer} --trace %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: RUNTIME ERROR: {{.*}}/explorer/testdata/tuple/fail_index_var.carbon:17: could not find `index: i32`

package ExplorerTest api;

fn Main() -> i32 {
  var x: auto = (0, 1);
  var index: i32 = 0;
  return x[index];
}
