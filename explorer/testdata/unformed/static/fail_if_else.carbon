// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

fn Foo() -> i32;

fn Main() -> i32 {
  var x: i32;
  if (Foo() == 0) {
    return 0;
  } else {
    // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/unformed/static/fail_if_else.carbon:[[@LINE+1]]: use of uninitialized variable x
    return x;
  }
}
