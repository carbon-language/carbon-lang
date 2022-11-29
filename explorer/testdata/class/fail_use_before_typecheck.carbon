// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

class C {
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/class/fail_use_before_typecheck.carbon:[[@LINE+1]]: incomplete type `class C` used in member access
  var n: if not C.WrapInStruct() then i32 else {.n: i32};
  fn WrapInStruct() -> bool { return true; }
}

fn Main() -> i32 {
  return 0;
}
