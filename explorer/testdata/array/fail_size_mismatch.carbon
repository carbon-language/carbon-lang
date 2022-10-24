// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

fn Main() -> i32 {
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/array/fail_size_mismatch.carbon:[[@LINE+1]]: type error in name binding: '(i32, i32, i32)' is not implicitly convertible to '[i32; 2]'
  var x: [i32; 2] = (0, 1, 2);
  return x[0];
}
