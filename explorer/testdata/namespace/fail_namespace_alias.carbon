// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

namespace N;
fn N.F() {}
// TODO: This should work.
// CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/namespace/fail_namespace_alias.carbon:[[@LINE+1]]: could not find `namespace N;`
alias M = N;

fn Main() -> i32 {
  M.F();
  return 0;
}
