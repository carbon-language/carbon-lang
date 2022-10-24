// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

interface HasThreeTypes {
  let A:! Type;
  let B:! Type;
  let C:! Type;
}

// CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/assoc_const/fail_incomplete_impl_1.carbon:[[@LINE+1]]: implementation doesn't provide a concrete value for interface HasThreeTypes.B
external impl i32 as HasThreeTypes where .A = i32 and .C = i32 {}

fn Main() -> i32 { return 0; }
