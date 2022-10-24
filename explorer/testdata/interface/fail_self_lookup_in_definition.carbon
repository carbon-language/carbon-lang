// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

interface Vector {
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/interface/fail_self_lookup_in_definition.carbon:[[@LINE+1]]: incomplete type `interface Vector` used in member access
  fn ScalarZero() -> Self.ScalarType;
  let ScalarType:! Type;
}

fn Main() -> i32 {
  return 0;
}
