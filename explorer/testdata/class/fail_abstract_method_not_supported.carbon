// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

base class C {
}

class D extends C {
  abstract fn Foo[self:Self]() -> i32 {
    return 1;
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/class/fail_abstract_method_not_supported.carbon:[[@LINE+1]]: Error declaring `Foo`: `abstract` methods are not yet supported.
  }
}

fn Main() -> i32 {
  let d: D = {};
  return 0;
}
