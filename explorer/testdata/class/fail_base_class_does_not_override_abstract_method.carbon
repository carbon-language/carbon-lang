// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

abstract class C {
  abstract fn Foo[self:Self]() -> i32;
}

base class D extends C {}

fn Main() -> i32 {
  let d: D = {.base = {}};
  // CHECK:STDERR: RUNTIME ERROR: {{.*}}/explorer/testdata/class/fail_base_class_does_not_override_abstract_method.carbon:[[@LINE+1]]: attempt to call function `Foo` that has not been defined
  d.Foo();
  return 0;
}
