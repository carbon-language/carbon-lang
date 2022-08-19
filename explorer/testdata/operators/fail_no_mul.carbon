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

class A {}

fn Main() -> i32 {
  var a: A = {};
  // CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/operators/fail_no_mul.carbon:[[@LINE+2]]: type error in `*`:
  // CHECK: could not find implementation of interface MulWith(U = i32) for class A
  a * 1;
  return 0;
}
