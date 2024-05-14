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

abstract class B {
  extend base: A;
}

class C {
  extend base: B;
// CHECK:STDERR: COMPILATION ERROR: fail_abstract_method_abstract_class.carbon:[[@LINE+1]]: Error declaring `C`: non abstract class should implement abstract method `Foo`.
}

fn Main() -> i32 {
  return 0;
}
