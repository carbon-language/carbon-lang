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

fn Foo() -> i32;

fn Main() -> i32 {
  var (x: i32, y: i32);
  while (Foo() == 0) {
    // CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/unformed/static/fail_while_body.carbon:[[@LINE+1]]: use of uninitialized variable x
    y = x;
  }
  return 1;
}
