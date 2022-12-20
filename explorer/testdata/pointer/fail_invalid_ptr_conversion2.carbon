// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

class A {}

fn Main() -> i32 {
  var a: i32 = 0;
  var b: i32* = &a;
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/pointer/fail_invalid_ptr_conversion2.carbon:[[@LINE+1]]: type error in name binding: 'i32*' is not implicitly convertible to 'class A*'
  var c: A* = b;
  return 1;
}
