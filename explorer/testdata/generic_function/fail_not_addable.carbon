// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | %{FileCheck-strict} %s
// RUN: %{not} %{explorer-trace} %s 2>&1 | %{FileCheck-allow-unmatched} %s
// AUTOUPDATE: %{explorer} %s

package ExplorerTest api;

fn id[T:! Type](x: T) -> T {
  // CHECK:COMPILATION ERROR: {{.*}}/explorer/testdata/generic_function/fail_not_addable.carbon:[[@LINE+2]]: type error in `+`:
  // CHECK:could not find implementation of interface AddWith(U = i32) for T:! Type
  return x + 0;
}

fn Main() -> i32 {
  return id(0);
}
