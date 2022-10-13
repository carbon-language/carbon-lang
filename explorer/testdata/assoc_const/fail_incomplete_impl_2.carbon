// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s | %{FileCheck-strict} %s
// RUN: %{not} %{explorer-trace} %s | %{FileCheck-allow-unmatched} %s
// AUTOUPDATE: %{explorer} %s

package ExplorerTest api;

interface HasThreeTypes {
  let A:! Type;
  let B:! Type;
  let C:! Type;
}

// CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/assoc_const/fail_incomplete_impl_2.carbon:[[@LINE+1]]: implementation doesn't provide a concrete value for (i32).(HasThreeTypes.B)
external impl i32 as HasThreeTypes where .A == i32 and .B == .C {}

fn Main() -> i32 { return 0; }
