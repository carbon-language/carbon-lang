// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{executable_semantics} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{not} %{executable_semantics} --trace %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{executable_semantics} %s
// CHECK: COMPILATION ERROR: {{.*}}/executable_semantics/testdata/function/fail_call_with_tuple.carbon:19: type error in call: '((i32, i32))' is not implicitly convertible to '(i32, i32)'

package ExecutableSemanticsTest api;

fn f(x: i32, y: i32) -> i32 { return x + y; }

fn Main() -> i32 {
  var xy: (i32, i32) = (1, 2);
  // should fail to type-check
  return f(xy);
}
