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

class D extends C {
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/class/fail_final_class_does_not_override_abstract_method:[[@LINE+1]]: Final class `D` must implement abstract method `Foo` from base class.
}

fn Main() -> i32 {
  return 0;
}
