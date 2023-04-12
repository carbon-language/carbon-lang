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

class B extends A {
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/class/fail_abstract_method_final_impl.carbon:[[@LINE+1]]: Error declaring `B`: final class should implement abstract method `Foo`.
}

fn Main() -> i32 {
  return 0;
}
