// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

base class C {
  virtual fn Foo[self:Self]() -> i32 {
    return 0;
  }
}

abstract class D extends C {
  abstract fn Foo[self:Self]() -> i32;
}

class E extends D {
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/class/fail_abstract_method_hides_base_implementation:[[@LINE+1]]: Final class `E` must implement abstract method `Foo` from base class.

}

fn Main() -> i32 {
  return 0;
}
