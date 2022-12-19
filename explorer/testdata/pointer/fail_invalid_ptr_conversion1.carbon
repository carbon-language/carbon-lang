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
  var a: A = {};
  var b: A* = &a;
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/pointer/fail_invalid_ptr_conversion1.carbon:[[@LINE+1]]: type error in name binding: 'class A*' is not implicitly convertible to 'i32'
  var c: i32 = b;
  return 1;
}
