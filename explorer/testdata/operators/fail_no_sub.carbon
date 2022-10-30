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
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/operators/fail_no_sub.carbon:[[@LINE+2]]: type error in `-`:
  // CHECK:STDERR: could not find implementation of interface SubWith(U = i32) for class A
  a - 1;
  return 0;
}
