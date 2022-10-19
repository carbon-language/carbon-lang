// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

interface Iface {
  // CHECK:STDERR: SYNTAX ERROR: {{.*}}/explorer/testdata/assoc_const/fail_anonymous.carbon:[[@LINE+1]]: syntax error, unexpected UNDERSCORE, expecting identifier
  let _:! Type;
}

fn Main() -> i32 { return 0; }
