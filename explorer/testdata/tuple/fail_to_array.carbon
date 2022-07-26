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

fn Main() -> i32 {
  var t: auto = (1, 2);
  // CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/tuple/fail_to_array.carbon:[[@LINE+1]]: type error in name binding: '(i32, i32)' is not implicitly convertible to '[i32; 3]'
  var a: [i32; 3] = t;
}
