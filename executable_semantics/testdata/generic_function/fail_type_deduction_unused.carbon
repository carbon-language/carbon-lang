// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{executable_semantics} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{not} %{executable_semantics} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{executable_semantics} %s
// CHECK: COMPILATION ERROR: {{.*}}/executable_semantics/testdata/generic_function/fail_type_deduction_unused.carbon:20: could not deduce type argument for type parameter T
// CHECK: in id(0)

package ExecutableSemanticsTest api;

fn id[T:! Type](x: i32) -> i32 {
  return x;
}

fn Main() -> i32 {
  return id(0);
}
