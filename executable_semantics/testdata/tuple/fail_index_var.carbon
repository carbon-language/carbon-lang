// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{executable_semantics} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{not} %{executable_semantics} --trace %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{executable_semantics} %s
// CHECK: RUNTIME ERROR: {{.*}}/executable_semantics/testdata/tuple/fail_index_var.carbon:17: could not find `index`

package ExecutableSemanticsTest api;

fn Main() -> i32 {
  var x: auto = (0, 1);
  var index: i32 = 0;
  return x[index];
}
