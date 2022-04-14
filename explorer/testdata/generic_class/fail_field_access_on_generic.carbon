// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{not} %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/generic_class/fail_field_access_on_generic.carbon:15: field access, unexpected T:! Type of non-interface type Type in a.x

package ExplorerTest api;

fn BadFieldAccess[T:! Type](a: T) -> T {
  return a.x;
}

fn Main() -> i32 {
  return BadFieldAccess(0);
}
