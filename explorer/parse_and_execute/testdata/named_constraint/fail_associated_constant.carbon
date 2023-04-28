// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

constraint X {
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/named_constraint/fail_associated_constant.carbon:[[@LINE+1]]: associated constant not permitted in named constraint
  let N:! type;
}

fn Main() -> i32 { return 0; }
