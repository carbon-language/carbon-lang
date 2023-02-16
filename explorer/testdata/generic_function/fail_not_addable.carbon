// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

fn id[T:! type](x: T) -> T {
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/generic_function/fail_not_addable.carbon:[[@LINE+2]]: type error in `+`:
  // CHECK:STDERR: could not find implementation of interface AddWith(U = i32) for T
  return x + 0;
}

fn Main() -> i32 {
  return id(0);
}
