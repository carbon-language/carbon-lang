// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{not} %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s

package ExplorerTest api;

fn AddInt(a: i32, b: i32) -> i32 {
  // CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/returned_var/fail_returned_var_type_mismatch.carbon:[[@LINE+1]]: type error in name binding: 'i32' is not implicitly convertible to 'Bool'
  returned var ret : Bool = a + b;
  return var;
}

fn Main() -> i32 {
  return AddInt(1, 2);
}
