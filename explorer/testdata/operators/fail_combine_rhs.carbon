// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

class C {}

// CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/operators/fail_combine_rhs.carbon:[[@LINE+1]]: expected a constraint in second operand of `&`, found class C
fn F[T:! Type & C]();

fn Main() -> i32 {
  return 0;
}
