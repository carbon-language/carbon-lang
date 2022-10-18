// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

base class C {
  fn F() {}
// CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/class/fail_base_class.carbon:[[@LINE+1]]: Class prefixes `base` and `abstract` are not supported yet
}

fn Main() -> i32 {
  return 0;
}
