// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

class C {
  fn PickType() -> type { return i32; }
  // This is invalid even though `PickType` is defined earlier, because
  // checking of member bodies is deferred until after the class is completed.
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/class/fail_member_call_before_typecheck.carbon:[[@LINE+1]]: attempt to call function `PickType` that has not been fully type-checked
  fn GetZero() -> PickType() { return 0; }
}

fn Main() -> i32 {
  return 0;
}
