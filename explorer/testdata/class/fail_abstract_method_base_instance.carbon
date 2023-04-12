// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

abstract class A {
    abstract fn Foo[self: Self]();
}

base class B extends A {

}

fn Main() -> i32 {
  // CHECK:STDERR: RUNTIME ERROR: {{.*}}/explorer/testdata/class/fail_abstract_method_base_instance.carbon:[[@LINE+1]]: Cannot instantiate base class `B`: abstract method `Foo` should be implemented.
  var b: B = { .base = {} };
  return 0;
}
