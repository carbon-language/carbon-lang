// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | %{FileCheck-strict} %s
// RUN: %{not} %{explorer-trace} %s 2>&1 | %{FileCheck-allow-unmatched} %s
// AUTOUPDATE: %{explorer} %s

package ExplorerTest api;

interface HasType {
  let T:! Type;
}

// CHECK:COMPILATION ERROR: {{.*}}/explorer/testdata/assoc_const/fail_overspecified_impl.carbon:[[@LINE+1]]: implementation provides multiple values for (.Self:! HasType).T: i32 and {.a: i32}
external impl i32 as HasType where .T == i32 and .T == {.a: i32} {}

fn Main() -> i32 { return 0; }
