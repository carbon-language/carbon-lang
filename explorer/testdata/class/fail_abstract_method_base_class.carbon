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

base class B {
  extend base: A;
// CHECK:STDERR: COMPILATION ERROR: fail_abstract_method_base_class.carbon:[[@LINE+1]]: Error declaring `B`: non abstract class should implement abstract method `Foo`.
}

fn Main() -> i32 {
  return 0;
}
