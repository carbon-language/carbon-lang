// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

base class C {
  virtual fn Foo[self:Self]() {}
}

class D extends C {
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/class/fail_redeclare_virtual_method.carbon:[[@LINE+1]]: Error declaring `Foo`: method is declared virtual in base class, use `impl` to override it.
  virtual fn Foo[self:Self]() {}
}

fn Main() -> i32 {
  let c: C = {};
  return 0;
}
