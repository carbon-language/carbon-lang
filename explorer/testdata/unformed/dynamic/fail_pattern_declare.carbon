// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

fn Main() -> i32 {
  var (x: i32, y: i32);
  x = 1;
  if (0 == 1) {
    y = 0;
  }
  // Static analysis thinks `y` may be formed, defer the check to run-time.
  // CHECK:STDERR: RUNTIME ERROR: {{.*}}/explorer/testdata/unformed/dynamic/fail_pattern_declare.carbon:[[@LINE+1]]: undefined behavior: access to uninitialized value Uninit<Placeholder<y>>
  return y;
}
