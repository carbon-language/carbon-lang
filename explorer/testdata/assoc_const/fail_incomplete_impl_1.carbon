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

// CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/assoc_const/fail_incomplete_impl_1.carbon:[[@LINE+1]]: implementation missing (i32).B; have constraint interface HasThreeTypes where i32 is interface HasThreeTypes and (i32).A == i32 and (i32).C == i32
external impl i32 as HasThreeTypes where .A == i32 and .C == i32 {}

fn Main() -> i32 { return 0; }
