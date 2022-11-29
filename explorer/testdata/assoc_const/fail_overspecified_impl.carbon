// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

interface HasType {
  let T:! Type;
}

// CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/assoc_const/fail_overspecified_impl.carbon:[[@LINE+3]]: multiple different rewrites for `(i32).(HasType.T)`:
// CHECK:STDERR:   i32
// CHECK:STDERR:   {.a: i32}
external impl i32 as HasType where .T = i32 and .T = {.a: i32} {}

fn Main() -> i32 { return 0; }
